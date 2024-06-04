using LabsPhi304;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.Reflection;

// path for model and images
var modelPath = @"d:\phi3\models\Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4";
var foggyDayImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "foggyday.png");
var petsMusicImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "petsmusic.png");
var ultraMugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "ultrarunningmug.png");

// load model and create processor
using Model model = new Model(modelPath);
using MultiModalProcessor processor = new MultiModalProcessor(model);
using var tokenizerStream = processor.CreateStream();
var tokenizer = new Tokenizer(model);

// define prompts
var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";


// write title
SpectreConsoleOutput.DisplayTitle($"C# - Phi-3v");

// user choice scenarios
var scenarios = SpectreConsoleOutput.SelectScenarios();
var scenario = scenarios[0];

// switch between the options for the selected scenario
// options can be file names with extension equal to ".png" or ".jpg"
// or the values "Type the image path to be analyzed" or "Type a question"
switch (scenario)
{
    case "foggyday.png":
    case "petsmusic.png":
    case "ultrarunningmug.png":
        var imagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", scenario);
        AnalizeImage(imagePath);
        break;
    case "Type the image path to be analyzed":
        scenario = SpectreConsoleOutput.AskForString("Type the image path to be analyzed");
        AnalizeImage(scenario);
        break;
    case "Type a question":        
        AnswerQuestion();
        break;
}
SpectreConsoleOutput.DisplayTitleH3("Done !");

void AnswerQuestion()
{
    var question = SpectreConsoleOutput.AskForString("Type a question");
    SpectreConsoleOutput.DisplayQuestion(question);
    SpectreConsoleOutput.DisplayAnswerStart("Phi-3");

    var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{question}<|end|><|assistant|>";
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
}

void AnalizeImage(string imagePath)
{
    var img = Images.Load(petsMusicImagePath);
    string userPrompt = "Describe the image, and return the string 'STOP' at the end.";
    var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|><|image_1|>{userPrompt}<|end|><|assistant|>";

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
}