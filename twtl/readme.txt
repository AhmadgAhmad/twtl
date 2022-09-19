<h3>>Setting up ANTLR4 Debugger<h3>

- Set up Antlr viewer extension in you vscode. 
- In the vscode Setting, specifically launch.json, add the following 
configuration: 
    {            "name": "Debug ANTLR4 grammar",
                "type": "antlr-debug",
                "request": "launch",
                "input": "/home/ahmad/Desktop/twtl/twtl/input.txt",
                "grammar": "/home/ahmad/Desktop/twtl/twtl/twtl.g4",
                "printParseTree": true,
                "visualParseTree": true
            }

