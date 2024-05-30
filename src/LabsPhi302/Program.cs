using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using System.Text.Json;

var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();
var builder = Kernel.CreateBuilder();
builder.AddAzureOpenAIChatCompletion(
    config["AZURE_OPENAI_MODEL"],
    config["AZURE_OPENAI_ENDPOINT"],
    config["AZURE_OPENAI_APIKEY"]);

var kernel = builder.Build();

var chat = kernel.GetRequiredService<IChatCompletionService>();


var systemPrompt = "You are an AI assistant that helps people find information.";

var history = new ChatHistory();
history.AddSystemMessage(systemPrompt);

var  userQ = "What is the capital of France?";
history.AddUserMessage(userQ);


var result = await chat.



// deserialize the object history to JSON format
var json = JsonSerializer.Serialize(history);
Console.WriteLine(json);


var response = "The capital of France is Paris.";
history.AddAssistantMessage(response);

json = JsonSerializer.Serialize(history);
Console.WriteLine(json);