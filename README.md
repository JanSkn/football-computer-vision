<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/janskn/football_computer_vision">
    <img src="https://github.com/JanSkn/football-computer-vision/assets/68644413/9ba661d9-992a-499a-89f4-946d3a99a359" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center">MatchVision</h3>

  <p align="center">
    Automated football match analysis using Computer Vision.
    <br />
    <a href="https://pylearn-ml.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/janskn/football_computer_vision/issues">Report Bug</a>
    ·
    <a href="https://github.com/janskn/football_computer_vision/issues">Request Feature</a>
  </p>

  <br />

</div>



<!-- ABOUT THE PROJECT -->
## About The Project

The project called *MatchVision* allows tracking the ball, referees and players, tracking them individually and assigning them to a team, displaying ball possession and estimating the camera movement. 

<br />

**Training**

The model was trained on Google Colab's servers with T4 GPU (~ 45 mins.)

It uses a custom trained YOLOv5 model.

Benefits:
- better ball tracking
- not tracking persons outside the pitch

<br />

**Structure**

The YOLO model detects the bounding boxes of the objects. They get stored in a dictionary along with the tracking ID, the team ID, the position etc.

The team detection is based on KMeans clustering.

To improve the ball detection, the ball position gets interpolated with Pandas.

The camera movement gets estimated with optical flow to adjust the object positions.

<br />

<img width="1101" alt="image" src="https://github.com/JanSkn/football-computer-vision/assets/68644413/09854bf6-39f5-4f7f-92c8-c8f438e7f7a1">


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

The source code was built with Python, mainly using Ultralytics, OpenCV and NumPy.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Requirements

Requirements can be found under `requirements.txt`.

```sh
pip install -r requirements.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Go to the root directory of the project and enter

```sh
streamlit run frontend/index.py
```

You can upload your own video or use a demo video. Everything is explained in the frontend.

<br />

Or run the `main.py` file.

<br />

<img width="1463" alt="image" src="https://github.com/JanSkn/football-computer-vision/assets/68644413/c5172932-b99f-4917-abfd-5a1005d20c9d">


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Please follow the [Contributing](.github/CONTRIBUTING.md) guidelines.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
