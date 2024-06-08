# face-recognition-attendance-recorder

## Usage

### Install the requirements:
```sh
pip install -r requirements.txt
```

### Split the weights file
Split the weights file into the CNN weights and the Similarity Score weights, then convert it (optional).
```sh
python convert_model.py [WEIGHT] [-f FORMATS]
```

- `weight`: File name of the `.h5` weights of the Siamese Network. Must be placed inside the `data/weights` directory.
- `-f`, `--formats`: Specify the output formats. Valid formats: `['onnx', 'h5']` Default to all valid formats.

### Run
Run the application using the following command:
```sh
python main.py [-r RUNTIME] [-w WEIGHT] [-t THRESHOLD] [-b BUFFER-SIZE] [-v]
```

- `-r`, `--runtime`: Inference runtime to use. Valid runtime: `['onnx', 'tf']` (default is `onnx`).
- `-w`, `--weight`: Directory containing the weight files for the Siamese network (default is the first directory in `data/weights` containing all valid weights).
- `-t`, `--threshold`: Threshold for face recognition confidence (default is `0.5`).
- `-b`, `--min-buffer-size`: Minimum buffer size for face recognition (default is `5`).
- `-v`, `--verbose`: Show verbose command line output.

#### Example

```sh
python main.py -r onnx -w siamese_weights -t 0.6 -b 10 -v
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
│       ├── siamese.weights
│       │   ├── cnn.onnx
│       │   ├── cnn.weights.h5
│       │   ├── simscore.onnx
│       │   └── simscore.weights.h5
│       └── siamese.weights.h5
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
