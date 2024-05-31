using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;


var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";
var userQuestion = "What is the capital of France?";
var assistantResponse = "The capital of France is Paris.";

var modelPath = @"D:\phi3\models\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32";

// create kernel
var builder = Kernel.CreateBuilder();
builder.add (modelPath, vocabPath) //AddOnnxRuntimeGenAIChatCompletion(modelPath: modelPath);
var kernel = builder.Build();

// create chat
var chat = kernel.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();

// run chat
while (true)
{
    Console.Write("Q: ");
    var userQ = Console.ReadLine();
    if (string.IsNullOrEmpty(userQ))
    {
        break;
    }
    history.AddUserMessage(userQ);

    Console.Write($"Phi3: ");
    var response = "";
    var result = chat.GetStreamingChatMessageContentsAsync(history);
    await foreach (var message in result)
    {
        Console.Write(message.Content);
        response += message.Content;
    }
    history.AddAssistantMessage(response);
    Console.WriteLine("");
}
