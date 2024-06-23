# NaiveBayesClassifier Documentation

## Overview

The `NaiveBayesClassifier` class implements a text classification algorithm based on the Naive Bayes theorem. It preprocesses text data, trains a model on labeled examples, evaluates its performance, supports model persistence, and provides visualization capabilities for interpreting important features.

### Features

- **Text Preprocessing**: Tokenization, stopword removal, stemming, and lemmatization.
- **Model Training**: Incremental training with text and category labels.
- **Classification**: Predicts the category of a given text using Naive Bayes probabilities.
- **Model Persistence**: Saves and loads trained models from JSON files.
- **Evaluation Metrics**: Computes accuracy, precision, recall, and F1-score.
- **Cross-validation**: Evaluates model performance using k-fold cross-validation.
- **Visualization**: Generates SVG charts to visualize important words based on TF-IDF scores.
- **Parallel Processing**: Utilizes worker threads for training and classification tasks.

## Installation

Install required npm packages if not already installed:
```bash
npm install fs lodash natural worker_threads d3-node
```

## Usage

### Initialization

```javascript
import fs from 'fs';
import _ from 'lodash';
import natural from 'natural';
import { TfIdf } from 'natural';
import { Worker, isMainThread, parentPort } from 'worker_threads';
import D3Node from 'd3-node';

const { WordTokenizer, PorterStemmer, WordNetLemmatizer, stopwords } = natural;

class NaiveBayesClassifier {
  constructor(stopWords = new Set(stopwords.words), n = 1) {
    this.categories = new Map();
    this.vocab = new Set();
    this.stopWords = stopWords;
    this.n = n; // For n-grams
  }
  // Methods...
}

export default NaiveBayesClassifier;
```

### Methods

#### `preprocess(text)`

Preprocesses the input text by tokenizing, filtering stopwords, stemming, and lemmatizing words.

**Usage:**
```javascript
const classifier = new NaiveBayesClassifier();
const processedWords = classifier.preprocess("Sample text to preprocess.");
```

#### `train(text, category)`

Trains the classifier with a text example labeled with a category.

**Usage:**
```javascript
classifier.train("Text to train with.", "category1");
```

#### `classify(text)`

Predicts the category of the given text.

**Usage:**
```javascript
const category = classifier.classify("Text to classify.");
```

#### `evaluate(testData)`

Evaluates the classifier's accuracy on a test dataset.

**Usage:**
```javascript
const accuracy = classifier.evaluate(testData);
```

#### `precisionRecallF1(testData)`

Computes precision, recall, and F1-score for multi-class classification.

**Usage:**
```javascript
const { precision, recall, f1 } = classifier.precisionRecallF1(testData);
```

#### `saveModel(filepath)`

Saves the trained model data to a JSON file.

**Usage:**
```javascript
classifier.saveModel('model.json');
```

#### `loadModel(filepath)`

Loads a previously saved model from a JSON file.

**Usage:**
```javascript
classifier.loadModel('model.json');
```

#### `crossValidate(data, k = 5)`

Performs k-fold cross-validation and returns the average accuracy.

**Usage:**
```javascript
const averageAccuracy = classifier.crossValidate(data, 5);
```

#### `visualizeImportantWords()`

Generates SVG charts using D3 to visualize top important words by TF-IDF score for each category.

**Usage:**
```javascript
classifier.visualizeImportantWords();
```

#### `runInWorker(data, callback)`

Utility function for running tasks in parallel using worker threads.

**Usage:**
```javascript
runInWorker(data, callback);
```

### Parallel Processing

The class supports parallel processing of training and classification tasks using Node.js worker threads. Ensure to handle messages and errors appropriately in the worker thread.

### Example Usage

```javascript
import NaiveBayesClassifier from './NaiveBayesClassifier';

const classifier = new NaiveBayesClassifier();

// Training
classifier.train("Text 1 to train with.", "category1");
classifier.train("Text 2 to train with.", "category2");
classifier.train("Text 3 to train with.", "category1");

// Classification
const category = classifier.classify("New text to classify.");

// Evaluation
const testData = [
  ["Test text 1", "category1"],
  ["Test text 2", "category2"]
];
const accuracy = classifier.evaluate(testData);

// Cross-validation
const data = [
  ["Training text 1", "category1"],
  ["Training text 2", "category2"],
  // More data...
];
const averageAccuracy = classifier.crossValidate(data, 5);

// Visualization
classifier.visualizeImportantWords();

// Save and Load Model
classifier.saveModel('model.json');
classifier.loadModel('model.json');
```

## Notes

- **Dependencies**: Requires `fs`, `lodash`, `natural`, `worker_threads`, and `d3-node`.
- **Thread Safety**: Ensure proper handling of worker threads and messages when using parallel processing.

---

This documentation provides comprehensive guidance on using the `NaiveBayesClassifier` class, covering initialization, methods, parameters, examples, and considerations for effective usage in JavaScript applications. Adjust paths and configurations as needed based on your project structure and requirements.
