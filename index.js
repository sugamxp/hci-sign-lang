let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
let previousPrediction = -1;
const wordList = document.getElementById('wordList');
const addWord = document.getElementById('addWord');
const newWord = document.getElementById('newWord');
const rephraseList = document.getElementById('list');
const langSelect = document.getElementById('langSelect');
let classes = ['Idle'];
let noOfImages = {
  Idle: 0,
};
let addExample;
async function app() {
  console.log('Loading mobilenet..');
  console.log(classifier);
  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  if (localStorage.getItem('classes') == null) {
    classesStorage = JSON.stringify(classes);
    localStorage.setItem('classes', classesStorage);
    noOfImagesStorage = JSON.stringify(noOfImages);
    localStorage.setItem('noOfImages', noOfImagesStorage);
  }

  if (localStorage.getItem('myModel') != null) {
    load();
    console.log('Load called');
  }

  await setupWebcam();

  // Reads an image from the webcam and associates it with a specific class
  // index.
  addExample = (word) => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const classId = classes.indexOf(word);
    noOfImages[word] += 1;
    noOfImagesStorage = JSON.stringify(noOfImages);
    localStorage.setItem('noOfImages', noOfImagesStorage);
    const btn = document.getElementById(word);
    btn.innerHTML = `${word} <span class="number">${noOfImages[word]}</span>`;
    const activation = net.infer(webcamElement, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
    save();
    console.log('Save done');
    console.log(classifier.getClassifierDataset());
  };

  // assign addExample event listener to the newly added buttons
  if (classes.length > 0) {
    for (word in classes) {
      console.log(classes[word]);
      const btn = document.getElementById(classes[word]);
      btn.addEventListener('click', () => {
        addExample(classes[word]);
      });
    }
  }

  // When clicking a button, add an example for that class.
  addWord.addEventListener('click', () => {
    const word = newWord.value;
    if (!word.length) {
      return;
    }
    if (word !== '' || classes.includes(word) == false) {
      const button = document.createElement('button');
      button.innerText = word;
      button.id = word;
      button.className = 'btn btn-primary mx-1';
      classes.push(word);
      noOfImages[word] = 0;
      noOfImagesStorage = JSON.stringify(noOfImages);
      classesStorage = JSON.stringify(classes);
      localStorage.setItem('classes', classesStorage);
      localStorage.setItem('noOfImages', noOfImagesStorage);
      button.addEventListener('click', () => addExample(word));
      newWord.value = '';
      wordList.appendChild(button);
    } else if (classes.includes(word)) {
      alert('Word already exists !! Please Enter new word !');
    } else {
      alert('Please Enter a valid Word');
    }
  });

  document
    .getElementById('Idle')
    .addEventListener('click', () => addExample('Idle'));
  document.getElementById('clear').addEventListener('click', () => {
    document.getElementById('console').innerText = '';
    // rephraseList.innerText = '';
    previousPrediction = -1;
  });
  document.getElementById('clearModel').addEventListener('click', () => {
    localStorage.removeItem('myModel');
    localStorage.removeItem('classes');
    localStorage.removeItem('noOfImages');
    setDefaultValue();
  });
  window.setInterval(async function () {
    console.log('running');
    if (classifier.getNumClasses() > 0) {
      // console.log(classes);
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);
      if (
        result.confidences[parseInt(result.label)] == 1 &&
        result.label != '0' &&
        result.label != previousPrediction
      ) {
        //  console.log(result);
        document
          .getElementById('console')
          .append(`${classes[parseInt(result.label)]} `);
        previousPrediction = result.label;
      }
    }
    await tf.nextFrame();
  }, 2000);
}

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia =
      navigator.getUserMedia ||
      navigatorAny.webkitGetUserMedia ||
      navigatorAny.mozGetUserMedia ||
      navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia(
        { video: true }, // To use rear camera replace { video: { facingMode: { exact: "environment" } } }
        (stream) => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata', () => resolve(), false);
        },
        (error) => reject()
      );
    } else {
      reject();
    }
  });
}

function save() {
  const dataset = classifier.getClassifierDataset();
  const datasetObj = {};
  Object.keys(dataset).forEach((key) => {
    const data = dataset[key].dataSync();
    // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...]
    // instead of object e.g {0:"0.1", 1:"-0.2"...}
    datasetObj[key] = Array.from(data);
  });
  const jsonStr = JSON.stringify(datasetObj);
  // console.log(jsonStr)
  // can be change to other source
  localStorage.setItem('myModel', jsonStr);
}

function setDefaultValue() {
  classes = ['Idle'];
  noOfImages = {
    Idle: 0,
  };
  classesStorage = JSON.stringify(classes);
  localStorage.setItem('classes', classesStorage);
  noOfImagesStorage = JSON.stringify(noOfImages);
  localStorage.setItem('noOfImages', noOfImagesStorage);
}

function load() {
  // can be change to other source
  const dataset = localStorage.getItem('myModel');
  classes = JSON.parse(localStorage.getItem('classes'));
  console.log(classes);
  noOfImages = JSON.parse(localStorage.getItem('noOfImages'));
  for (word in classes) {
    console.log(word);
    if (word != 0) {
      const button = document.createElement('button');
      button.id = classes[word];
      button.className = 'btn btn-primary mx-1';
      button.innerHTML = `${classes[word]} <span class="number">${
        noOfImages[classes[word]]
      }</span>`;
      wordList.appendChild(button);
      // button.addEventListener('click', () => addExample(classes[word]));
    }
  }
  const tensorObj = JSON.parse(dataset);
  // covert back to tensor
  console.log(tensorObj);
  Object.keys(tensorObj).forEach((key) => {
    console.log(key);
    tensorObj[key] = tf.tensor(tensorObj[key], [
      tensorObj[key].length / 1024,
      1024,
    ]);
  });
  console.log(tensorObj);
  classifier.setClassifierDataset(tensorObj);
}
app();
