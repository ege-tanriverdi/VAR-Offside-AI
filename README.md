> This project is a prototype built for learning and experimentation. It is not accurate enough for real-world officiating decisions.

# Offside Line Prototype

This project is an experimental computer vision prototype for detecting players in football footage and estimating a reference offside line.

## Project Goal
The goal of this project is to explore how computer vision techniques can be used to analyze football scenes and visualize an estimated offside line.

## Current Status
This project is currently a prototype and should not be considered a reliable real-world offside decision system.

## Features
- player detection from football footage
- basic team color grouping
- reference line estimation
- output video generation

## Technologies Used
- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- scikit-learn

## Limitations
- team classification is not fully reliable
- camera angle and perspective can affect results
- defender selection logic is simplified
- not suitable for real match decisions

## Future Improvements
- improve defender selection logic
- add player tracking across frames
- improve team classification consistency
- test with cleaner datasets and multiple camera angles
- separate the codebase into modules

## Project Structure

```bash
data/
  input/      # put input videos here
  output/     # processed videos are saved here
src/
  main.py