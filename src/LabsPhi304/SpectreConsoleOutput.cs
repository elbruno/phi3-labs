#pragma warning disable SKEXP0001, SKEXP0003, SKEXP0010, SKEXP0011, SKEXP0050, SKEXP0052
using Spectre.Console;

namespace LabsPhi304;

public static class SpectreConsoleOutput
{
    public static void DisplayTitle(string title = ".NET - Phi-3 Vision Sample")
    {
        AnsiConsole.Write(new FigletText(title).Centered().Color(Color.Purple));
    }

    public static void DisplayTitleH2(string subtitle)
    {
        AnsiConsole.MarkupLine($"[bold][blue]=== {subtitle} ===[/][/]");
        AnsiConsole.MarkupLine($"");
    }

    public static void DisplayTitleH3(string subtitle)
    {
        AnsiConsole.MarkupLine($"[bold]>> {subtitle}[/]");
        AnsiConsole.MarkupLine($"");
    }

    public static void DisplayQuestion(string question)
    {
        AnsiConsole.MarkupLine($"[bold][blue]>>Q: {question}[/][/]");
        AnsiConsole.MarkupLine($"");
    }
    public static void DisplayAnswerStart(string answerPrefix)
    {
        AnsiConsole.Markup($"[bold][blue]>> {answerPrefix}:[/][/]");
    }

    public static void DisplayFilePath(string prefix, string filePath)
    {
        var path = new TextPath(filePath);

        AnsiConsole.Markup($"[bold][blue]>> {prefix}: [/][/]");
        AnsiConsole.Write(path);
        AnsiConsole.MarkupLine($"");
    }

    public static void DisplaySubtitle(string prefix, string content)
    {
        AnsiConsole.Markup($"[bold][blue]>> {prefix}: [/][/]");
        AnsiConsole.WriteLine(content);
        AnsiConsole.MarkupLine($"");
    }



    public static int AskForNumber(string question)
    {
        var number = AnsiConsole.Ask<int>(@$"[green]{question}[/]");
        return number;
    }

    public static string AskForString(string question)
    {
        var response = AnsiConsole.Ask<string>(@$"[green]{question}[/]");
        return response;
    }

    public static List<string> SelectScenarios()
    {
        // Ask for the user's favorite fruits
        var scenarios = AnsiConsole.Prompt(
            new MultiSelectionPrompt<string>()
                .Title("Select the [green]Phi 3 Vision scenarios[/] to run?")
                .PageSize(10)
                .Required(true)
                .MoreChoicesText("[grey](Move up and down to reveal more scenarios)[/]")
                .InstructionsText(
                    "[grey](Press [blue]<space>[/] to toggle a scenario, " +
                    "[green]<enter>[/] to accept)[/]")
                .AddChoiceGroup("Select an image to be analuyzed", new[]
                    {"foggyday.png","foggydaysmall.png","petsmusic.png","ultrarunningmug.png",
                    })
                .AddChoices( new[] { 
                    "Type the image path to be analyzed",
                    "Type a question"
                    })
                );
        return scenarios;
    }
}