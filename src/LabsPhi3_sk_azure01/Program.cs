#pragma warning disable SKEXP0010 
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();
var modelId = config["AZURE_AI_PHI3_MODEL"];
var endPoint = config["AZURE_AI_PHI3_ENDPOINT"];
var apiKey = config["AZURE_AI_PHI3_APIKEY"];

// create kernel
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(modelId, new Uri(endPoint), apiKey);
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
