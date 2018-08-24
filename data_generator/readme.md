# Data Generator Readme
This jar is used to collect the game log data for the workshop. 

## Config File
The config file contains all of the data needed to run the experiement. It has 4 mandatory fields and one optional field
- `id VALUE` - Here `VALUE` is a unique identifier for the Human or AI player
- `game VALUE` - Here `VALUE` needs to be the GVGAI game AI (e.g. `0` for Aliens)
- `level VALUE` - Here `VALUE` is which level (0-4) to run
- `skill VALUE` - Here `VALUE` is the skill of the AI/Human (0 - Low, 1 - Medium, 2- High)
- `ai` - This line is required to run an AI again, remove if running for a human player

## AI IDs
To select which AI agent to use set the `id` values to one of the following values:
- `0` - OneStepLookAhead - Skill: Low
- `1` - sampleGA - Skill:Low
- `2` - CatLinux - Skill: Medium
- `3` - SampleMCTS - Skill: Medium
- `4` - YOLOBot - Skill: High
- `5` - YBCriber - Skill: High

## Game IDs
Three games are used in the workshop, Aliens, Frogs and SeaQuest. The game IDs are:
- `0` - Aliens
- `42` - Frogs
- `77` - SeaQuest

## Running the Jar
To run the jar navigate to this directory in your terminal of choice, make sure the config is configured correctly and run `java -jar GVGAI.jar`.