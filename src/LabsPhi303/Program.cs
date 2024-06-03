using System;
using System.Reflection.Emit;
using System.Reflection;
using Microsoft.ML.OnnxRuntimeGenAI;


var modelPath = @"d:\phi3\models\Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4";
var model = new Model(modelPath);
var tokenizer = new Tokenizer(model);

var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

var imageUrl = @"https://github.com/elbruno/gpt4ol-sk-csharp/blob/main/imgs/rpi5.png?raw=true";
var describeImageQuestion = " ";

// show phi3 response
Console.Write("Phi3: ");
var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{describeImageQuestion}<|end|><|assistant|>";
var tokens = tokenizer.Encode(fullPrompt);

var generatorParams = new GeneratorParams(model);
generatorParams.SetSearchOption("max_length", 2048);
generatorParams.SetSearchOption("past_present_share_buffer", false);
generatorParams.SetInputSequences(tokens);

var generator = new Generator(model, generatorParams);
while (!generator.IsDone())
{
    generator.ComputeLogits();
    generator.GenerateNextToken();
    var outputTokens = generator.GetSequence(0);
    var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
    var output = tokenizer.Decode(newToken);
    Console.Write(output);
}
Console.WriteLine();
