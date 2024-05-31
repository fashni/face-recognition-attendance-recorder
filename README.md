# face-recognition-attendance-recorder

## Usage

Install the requirements:
```sh
pip install -r requirements.txt
```

Run the application using the following command:
```sh
python main.py [-w WEIGHT] [-t THRESHOLD]
```

- `-w`, `--weight`: Filename of the weight file for the Siamese network (default is the first `.h5` file in `data/weights`).
- `-t`, `--threshold`: Threshold for face recognition confidence (default is `0.5`).

## Directory Structure

```
.
├── data
│   ├── assets
│   │   └── overlay.png
│   ├── attendance
│   ├── known_faces
│   │   └── person1.jpg
│   └── weights
│       └── model.h5
├── main.py
├── requirements.txt
└── siamese_network.py
```

## Acknowledgements

This application uses the Haar Cascade Classifier for face detection and a custom [Siamese network](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/) for face recognition. Special thanks to the open-source community for providing the necessary libraries and tools.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
