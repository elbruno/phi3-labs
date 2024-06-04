using Microsoft.ML.OnnxRuntimeGenAI;

// path for model and images
var modelPath = @"d:\phi3\models\Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4";

var foggyDayImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "foggyday.png");
var petsMusicImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "petsmusic.png");
var img = Images.Load(petsMusicImagePath);

// define prompts
var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";
string userPrompt = "Describe the image, and return the string 'STOP' at the end.";
var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|><|image_1|>{userPrompt}<|end|><|assistant|>";

// load model and create processor
using Model model = new Model(modelPath);
using MultiModalProcessor processor = new MultiModalProcessor(model);
using var tokenizerStream = processor.CreateStream();

// create the input tensor with the prompt and image
Console.WriteLine("Full Prompt: " + fullPrompt);
Console.WriteLine("Start processing image and prompt ...");
var inputTensors = processor.ProcessImages(fullPrompt, img);
using GeneratorParams generatorParams = new GeneratorParams(model);
generatorParams.SetSearchOption("max_length", 3072);
generatorParams.SetInputs(inputTensors);

// generate response
Console.WriteLine("Generating response ...");
using var generator = new Generator(model, generatorParams);
while (!generator.IsDone())
{
    generator.ComputeLogits();
    generator.GenerateNextToken();
    var seq = generator.GetSequence(0)[^1];
    Console.Write(tokenizerStream.Decode(seq));
}

Console.WriteLine("");
Console.WriteLine("Done!");