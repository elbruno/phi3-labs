# Labs using Phi-3 and Phi-3-Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](/LICENSE)
[![Twitter: elbruno](https://img.shields.io/twitter/follow/elbruno.svg?style=social)](https://twitter.com/elbruno)
![GitHub: elbruno](https://img.shields.io/github/followers/elbruno?style=social)

Welcome to the Phi-3 samples using C#. This repository contains a set of demo projects that showcases how to integrate the powerful different versions of Phi-3 models in a .NET environment.

## Prerequisites

Before running the sample, ensure you have the following installed:
- **.NET 8**: Make sure you have the latest version of .NET installed on your machine.
- **(Optional) Visual Studio or Visual Studio Code**: You will need an IDE or code editor capable of running .NET projects. Visual Studio or Visual Studio Code are recommended.
- Using git, clone locally one of the available Phi-3 versions. 

    Download the **phi3-mini-4k-instruct-onnx** model to your local machine:
    ```bash
    # navigate to the folder to store the models
    # download phi3-mini-4k-instruct-onnx
    cd c:\phi3\models
    git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
    ```
Important: The current demos are designed to use the ONNX versions of the model. Clone the following modes. Currently there is no ONNX version available for the **Phi-3-vision-128k-instruct** model 
 ![Download only ONNX models](./img/10DownloadOnnx.png)


## About the Samples

The main solution have several sample projects that demonstrates the capabilities of the Phi-3 models.

| Project Name | Description | Location |
| ------------ | ----------- | -------- |
| LabsPhi301    | This is a sample project that uses a local phi3 model to ask a question | .\src\LabsPhi301\ |
| LabsPhi302    | This is a sample project to create a coherent chat history using the local model. | .\src\LabsPhi302\ |


## How to Run the Project

To run the project, follow these steps:
1. Clone the repository to your local machine.

1. Open a terminal and navigate to the desired project. 
    ```bash
    cd .\src\LabsPhi301\
    ```

1. Run the project with the command
    ```bash
    dotnet run
    ```

1.  The sample project ask for a user input and replies using the local mode. The running demo is similar to this one:

    ![Chat running demo](/imgs/20chatdemo.gif)



## Author

👤 **Bruno Capuano**

* Website: https://elbruno.com
* Twitter: [@elbruno](https://twitter.com/elbruno)
* Github: [@elbruno](https://github.com/elbruno)
* LinkedIn: [@elbruno](https://linkedin.com/in/elbruno)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

Feel free to check [issues page](https://github.com/elbruno/gpt4ol-sk-csharp//issues).

## Show your support

Give a ⭐️ if this project helped you!


## 📝 License

Copyright &copy; 2024 [Bruno Capuano](https://github.com/elbruno).

This project is [MIT](/LICENSE) licensed.

***