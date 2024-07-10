#pragma warning disable SKEXP0010 
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();
var modelId = config["AZURE_AI_PHI3V_MODEL"];
var endPoint = config["AZURE_AI_PHI3V_ENDPOINT"];
var apiKey = config["AZURE_AI_PHI3V_APIKEY"];

// create kernel
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(modelId, new Uri(endPoint), apiKey);
var kernel = builder.Build();

// create chat
var chat = kernel.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();

var petsMusicImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "petsmusic.png");

// create chat collection items
var collectionItems = new ChatMessageContentItemCollection
{
    new TextContent("What's in the image?"),
    //new ImageContent( new Uri(petsMusicImagePath))
    //new ImageContent( new Uri("https://github.com/elbruno/gpt4o-labs-csharp/tree/main/src/GPT4o_AOAI_lab02/imgs/foggyday.png?raw=true"))
    new ImageContent(File.ReadAllBytes(petsMusicImagePath))
};
history.AddUserMessage(collectionItems);

Console.Write($"Phi3: ");
var result = await chat.GetChatMessageContentsAsync(history);
Console.WriteLine(result[^1].Content);

Console.WriteLine("");
