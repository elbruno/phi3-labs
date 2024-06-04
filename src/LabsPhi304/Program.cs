using feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;


var systemPrompt = "You are an AI assistant that helps people answering their questions. Answer questions using a direct style. Do not share more information that the requested by the users.";

var modelPath = @"d:\phi3\models\Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4";

// create kernel
var builder = Kernel.CreateBuilder();
builder.AddOnnxRuntimeGenAIChatCompletion(modelPath: modelPath);
var kernel = builder.Build();

// create chat
var chat = kernel.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();


//new Uri("https://github.com/elbruno/gpt4ol-sk-csharp/blob/main/imgs/rpi5.png?raw=true"))
// create a local uri for the image in the current path + "imgs\petsmusic.jpg"
var imagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "petsmusic.jpg");
var imageUri = new Uri(imagePath);
var collectionItems = new ChatMessageContentItemCollection
{
    new ImageContent(imageUri),
    new TextContent("What is shown in this image"),
    new TextContent("What's the capital of Germany?"),
};

//var imagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "ultrarunningmug.png");
//var imageBytes = File.ReadAllBytes(imagePath);
//var collectionItems = new ChatMessageContentItemCollection
//{
//    new ImageContent(imageBytes),
//    new TextContent("What is shown in this image"),
//};


history.AddUserMessage(collectionItems);


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
