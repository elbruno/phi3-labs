using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Services;
using philabs.SemanticKernel.Connectors.OnnxRuntimeGenAI.Models;

namespace philabs.SemanticKernel.Connectors.OnnxRuntimeGenAI;

/// <summary>
/// Represents a chat completion service using OnnxRuntimeGenAI.
/// </summary>
public sealed class OnnxRuntimeGenAIChatCompletionService : IChatCompletionService
{
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly MultiModalProcessor _processor;
    private readonly TokenizerStream _tokenizerStream;
    private Dictionary<string, object?> AttributesInternal { get; } = new();

    /// <summary>
    /// Initializes a new instance of the OnnxRuntimeGenAIChatCompletionService class.
    /// </summary>
    /// <param name="modelPath">The generative AI ONNX model path for the chat completion service.</param>
    /// <param name="loggerFactory">Optional logger factory to be used for logging.</param>
    public OnnxRuntimeGenAIChatCompletionService(
        string modelPath,
        ILoggerFactory? loggerFactory = null)
    {
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);

        if (modelPath.Contains("vision"))
        {
            _processor = new MultiModalProcessor(_model);
            _tokenizerStream = _processor.CreateStream();
        }

        this.AttributesInternal.Add(AIServiceExtensions.ModelIdKey, _tokenizer);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, object?> Attributes => this.AttributesInternal;

    /// <inheritdoc />
    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var result = new StringBuilder();

        await foreach (var content in RunInferenceAsync(chatHistory, executionSettings, cancellationToken))
        {
            result.Append(content);
        }

        return new List<ChatMessageContent>
        {
            new(
                role: AuthorRole.Assistant,
                content: result.ToString())
        };
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        await foreach (var content in RunInferenceAsync(chatHistory, executionSettings, cancellationToken))
        {
            yield return new StreamingChatMessageContent(AuthorRole.Assistant, content);
        }
    }

    private async IAsyncEnumerable<string> RunInferenceAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings, CancellationToken cancellationToken)
    {
        OnnxRuntimeGenAIPromptExecutionSettings onnxRuntimeGenAIPromptExecutionSettings = OnnxRuntimeGenAIPromptExecutionSettings.FromExecutionSettings(executionSettings);

        var promptResult = GetPrompt(chatHistory, onnxRuntimeGenAIPromptExecutionSettings);

        Generator generator;

        var generatorParams = new GeneratorParams(_model);
        ApplyPromptExecutionSettings(generatorParams, onnxRuntimeGenAIPromptExecutionSettings);

        if (!promptResult.ImageFound)
        {
            var tokens = _tokenizer.Encode(promptResult.Prompt);
            generatorParams.SetInputSequences(tokens);
        }
        else
        {
            var img = Images.Load(promptResult.ImagePath);
            var inputTensors = _processor.ProcessImages(promptResult.Prompt, img);
            generatorParams.SetSearchOption("max_length", 3072);
            generatorParams.SetInputs(inputTensors);
        }

        generator = new Generator(_model, generatorParams);


        if (generator is not null)
            while (!generator.IsDone())
            {
                cancellationToken.ThrowIfCancellationRequested();

                yield return await Task.Run(() =>
                {
                    generator.ComputeLogits();
                    generator.GenerateNextToken();

                    var outputTokens = generator.GetSequence(0);
                    var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
                    var output = _tokenizer.Decode(newToken);
                    return output;
                }, cancellationToken);
            }

    }

    private PromptBuilderResult GetPrompt(ChatHistory chatHistory, OnnxRuntimeGenAIPromptExecutionSettings onnxRuntimeGenAIPromptExecutionSettings)
    {
        var result = new PromptBuilderResult();
        var promptBuilder = new StringBuilder();
        foreach (var message in chatHistory)
        {
            promptBuilder.Append($"<|{message.Role}|>\n{message.Content}");

            // process sub items
            foreach (var item in message.Items)
            {
                if (item is ImageContent imageContent)
                {
                    result.ImageFound = true;
                    promptBuilder.Append($"<|image_1|>");

                    var imageItem = item as ImageContent;

                    if (imageItem?.Data != null)
                    {
                        result.ImageBytes = imageItem.Data.Value.ToArray();
                    }
                    else if (imageItem?.Uri != null)
                    {
                        result.Uri = imageItem.Uri;
                    }
                    break;
                }
            }

        }
        promptBuilder.Append($"<|end|>\n<|assistant|>");

        result.Prompt = promptBuilder.ToString();

        return result;
    }

    private void ApplyPromptExecutionSettings(GeneratorParams generatorParams, OnnxRuntimeGenAIPromptExecutionSettings onnxRuntimeGenAIPromptExecutionSettings)
    {
        generatorParams.SetSearchOption("top_p", onnxRuntimeGenAIPromptExecutionSettings.TopP);
        generatorParams.SetSearchOption("top_k", onnxRuntimeGenAIPromptExecutionSettings.TopK);
        generatorParams.SetSearchOption("temperature", onnxRuntimeGenAIPromptExecutionSettings.Temperature);
        generatorParams.SetSearchOption("repetition_penalty", onnxRuntimeGenAIPromptExecutionSettings.RepetitionPenalty);
        generatorParams.SetSearchOption("past_present_share_buffer", onnxRuntimeGenAIPromptExecutionSettings.PastPresentShareBuffer);
        generatorParams.SetSearchOption("num_return_sequences", onnxRuntimeGenAIPromptExecutionSettings.NumReturnSequences);
        generatorParams.SetSearchOption("no_repeat_ngram_size", onnxRuntimeGenAIPromptExecutionSettings.NoRepeatNgramSize);
        generatorParams.SetSearchOption("min_length", onnxRuntimeGenAIPromptExecutionSettings.MinLength);
        generatorParams.SetSearchOption("max_length", onnxRuntimeGenAIPromptExecutionSettings.MaxLength);
        generatorParams.SetSearchOption("length_penalty", onnxRuntimeGenAIPromptExecutionSettings.LengthPenalty);
        generatorParams.SetSearchOption("early_stopping", onnxRuntimeGenAIPromptExecutionSettings.EarlyStopping);
        generatorParams.SetSearchOption("do_sample", onnxRuntimeGenAIPromptExecutionSettings.DoSample);
        generatorParams.SetSearchOption("diversity_penalty", onnxRuntimeGenAIPromptExecutionSettings.DiversityPenalty);
    }
}
