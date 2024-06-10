//    Copyright (c) 2024
//    Author      : Bruno Capuano
//    Change Log  :
//
//    The MIT License (MIT)
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
//    all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//    THE SOFTWARE.

using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

// path for model and images
var modelPath = @"d:\phi3\models\Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4";

// create kernel
var builder = Kernel.CreateBuilder();
builder.AddOnnxRuntimeGenAIChatCompletion(modelPath: modelPath);
var kernel = builder.Build();

// create chat
var chat = kernel.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();

var petsMusicImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "petsmusic.png");

// create chat collection items
var collectionItems = new ChatMessageContentItemCollection
{
    new TextContent("What's in the image?"),
    new ImageContent( new Uri(petsMusicImagePath))
    //new ImageContent( new Uri("https://github.com/elbruno/gpt4o-labs-csharp/tree/main/src/GPT4o_AOAI_lab02/imgs/foggyday.png?raw=true"))
    //new ImageContent(File.ReadAllBytes(petsMusicImagePath))
};
history.AddUserMessage(collectionItems);

Console.Write($"Phi3: ");
var result = await chat.GetChatMessageContentsAsync(history);
Console.WriteLine(result[^1].Content);

Console.WriteLine("");
