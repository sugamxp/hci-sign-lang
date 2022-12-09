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

  const addExample = (word) => {

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

  if (classes.length > 0) {
    for (word in classes) {
      console.log(classes[word]);
      const btn = document.getElementById(classes[word]);
      btn.addEventListener('click', () => {
        addExample(classes[word]);
      });
    }
  }

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
      const activation = net.infer(webcamElement, 'conv_preds');
      const result = await classifier.predictClass(activation);
      if (
        result.confidences[parseInt(result.label)] == 1 &&
        result.label != '0' &&
        result.label != previousPrediction
      ) {
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
        { video: true }, 
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
  const dataset_object = {};
  Object.keys(dataset).forEach((key) => {
    const data = dataset[key].dataSync();
    dataset_object[key] = Array.from(data);
  });
  const jsonStr = JSON.stringify(dataset_object);
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
    }
  }
  const tensorObj = JSON.parse(dataset);
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
