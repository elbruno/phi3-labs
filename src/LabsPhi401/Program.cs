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
using System.Reflection.Emit;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

//var modelPath = @"D:\phi3\models\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32";
var modelPath = @"d:\phi3\models\Phi-3.5-mini-instruct-onnx\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4\phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx";
var session = new InferenceSession(modelPath);

var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

// chat start
Console.WriteLine(@"Ask your question. Type an empty string to Exit.");

// chat loop
while (true)
{
    // Get user question
    Console.WriteLine();
    Console.Write(@"Q: ");
    //var userQ = Console.ReadLine();
    var userQ = "2+2";
    if (string.IsNullOrEmpty(userQ))
    {
        break;
    }

    // show phi3 response
    Console.Write("Phi3: ");


    // create input tensor (nlp example)
    var inputOrtValue = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, new long[] { 1, 1 });
    inputOrtValue.StringTensorSetElementAt(userQ, 0);

    // Create input data for session. Request all outputs in this case.
    var inputs = new Dictionary<string, OrtValue>
            {
                { "input", inputOrtValue }
            };

    var runOptions = new RunOptions();

    // We are getting a sequence of maps as output. We are interested in the first element (map) of the sequence.
    // That result is a Sequence of Maps, and we only need the first map from there.
    var outputs = session.Run(runOptions, inputs, session.OutputNames);    
    Debug.Assert(outputs.Count > 0, "Expecting some output");

    // We want the last output, which is the sequence of maps
    var lastOutput = outputs[outputs.Count - 1];

    // Optional code to check the output type
    {
        var outputTypeInfo = lastOutput.GetTypeInfo();
        Debug.Assert(outputTypeInfo.OnnxType == OnnxValueType.ONNX_TYPE_SEQUENCE, "Expecting a sequence");

        var sequenceTypeInfo = outputTypeInfo.SequenceTypeInfo;
        Debug.Assert(sequenceTypeInfo.ElementType.OnnxType == OnnxValueType.ONNX_TYPE_MAP, "Expecting a sequence of maps");
    }

    var elementsNum = lastOutput.GetValueCount();
    Debug.Assert(elementsNum > 0, "Expecting a non empty sequence");

    // Get the first map in sequence
    var firstMap = lastOutput.GetValue(0, OrtAllocator.DefaultInstance);

    // Optional code just checking
    {
        // Maps always have two elements, keys and values
        // We are expecting this to be a map of strings to floats
        var mapTypeInfo = firstMap.GetTypeInfo().MapTypeInfo;
        Debug.Assert(mapTypeInfo.KeyType == TensorElementType.String, "Expecting keys to be strings");
        Debug.Assert(mapTypeInfo.ValueType.OnnxType == OnnxValueType.ONNX_TYPE_TENSOR, "Values are in the tensor");
        Debug.Assert(mapTypeInfo.ValueType.TensorTypeAndShapeInfo.ElementDataType == TensorElementType.Float, "Result map value is float");
    }

    var inferenceResult = new Dictionary<string, float>();
    // Let use the visitor to read map keys and values
    // Here keys and values are represented with the same number of corresponding entries
    // string -> float
    firstMap.ProcessMap((keys, values) => {
        // Access native buffer directly
        var valuesSpan = values.GetTensorDataAsSpan<float>();

        var entryCount = (int)keys.GetTensorTypeAndShape().ElementCount;
        inferenceResult.EnsureCapacity(entryCount);
        for (int i = 0; i < entryCount; ++i)
        {
            inferenceResult.Add(keys.GetStringElement(i), valuesSpan[i]);
        }
    }, OrtAllocator.DefaultInstance);


    Console.WriteLine(inferenceResult.Aggregate("", (current, next) => current + $"{next.Key} "));

    //// Return the inference result as json.
    //return new JsonResult(inferenceResult);

    Console.WriteLine();
}
