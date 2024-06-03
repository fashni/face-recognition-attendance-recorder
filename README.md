# face-recognition-attendance-recorder

## Usage

Install the requirements:
```sh
pip install -r requirements.txt
```

Run the application using the following command:
```sh
python main.py [-w WEIGHT] [-t THRESHOLD] [-b BUFFER-SIZE] [-v]
```

- `-w`, `--weight`: Filename of the weight file for the Siamese network (default is the first `.h5` file in `data/weights`).
- `-t`, `--threshold`: Threshold for face recognition confidence (default is `0.5`).
- `-b`, `--min-buffer-size`: Minimum buffer size for face recognition (default is `10`).
- `-v`, `--verbose`: Show verbose command line output.

### Example

```sh
python attendance_recorder.py -w siamese_weights.h5 -t 0.6 -b 10 -v
```

## Directory Structure

```
.
├── data
│   ├── assets
│   │   └── overlay.png
│   ├── records
│   │   └── today.csv
│   ├── known_faces
│   │   └── person1.jpg
│   └── weights
│       └── siamese_weight.h5
├── main.py
├── requirements.txt
├── siamese_network.py
└── utils.py
```
- `data/weights`: Directory for storing the pre-trained weights.
- `data/records`: Directory for storing attendance records.
- `data/known_faces`: Directory for storing images of known faces.
- `data/assets`: Directory for storing GUI assets.

## Acknowledgements

This application uses the Haar Cascade Classifier for face detection and a custom [Siamese network](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/) for face recognition. Special thanks to the open-source community for providing the necessary libraries and tools.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
