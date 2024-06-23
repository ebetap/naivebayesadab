import fs from 'fs';
import _ from 'lodash';
import natural from 'natural';
import { TfIdf } from 'natural';

const defaultStopWords = new Set(natural.stopwords);
const stemmer = natural.PorterStemmer;
const lemmatizer = new natural.WordNetLemmatizer();

class NaiveBayesClassifier {
  constructor(stopWords = defaultStopWords, n = 1) {
    this.categories = {};
    this.vocab = new Set();
    this.stopWords = stopWords;
    this.n = n; // For n-grams
  }

  preprocess(text) {
    const words = text
      .toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => !this.stopWords.has(word))
      .map(word => lemmatizer.lemmatize(stemmer.stem(word)));

    if (this.n > 1) {
      return _.flatMap(words, (word, i) => {
        if (i <= words.length - this.n) {
          return [words.slice(i, i + this.n).join(' ')];
        }
        return [];
      });
    }

    return words;
  }

  train(text, category) {
    const words = this.preprocess(text);
    if (!this.categories[category]) {
      this.categories[category] = { total: 0, wordCount: {}, tfidf: new TfIdf() };
    }

    this.categories[category].tfidf.addDocument(words.join(' '));

    words.forEach(word => {
      this.categories[category].wordCount[word] = (this.categories[category].wordCount[word] || 0) + 1;
      this.vocab.add(word);
      this.categories[category].total++;
    });
  }

  classify(text) {
    const words = this.preprocess(text);
    const scores = {};

    Object.keys(this.categories).forEach(category => {
      scores[category] = Math.log((this.categories[category].total + 1) / (this.totalExamples() + Object.keys(this.categories).length));

      words.forEach(word => {
        const wordCount = this.categories[category].wordCount[word] || 0;
        scores[category] += Math.log((wordCount + 1) / (this.categories[category].total + this.vocab.size));
      });
    });

    return _.maxBy(Object.keys(scores), category => scores[category]);
  }

  totalExamples() {
    return Object.values(this.categories).reduce((total, category) => total + category.total, 0);
  }

  saveModel(filepath) {
    try {
      const modelData = {
        categories: this.categories,
        vocab: Array.from(this.vocab)
      };
      fs.writeFileSync(filepath, JSON.stringify(modelData));
    } catch (error) {
      console.error('Error saving model:', error);
    }
  }

  loadModel(filepath) {
    try {
      const modelData = JSON.parse(fs.readFileSync(filepath, 'utf8'));
      this.categories = modelData.categories;
      this.vocab = new Set(modelData.vocab);
    } catch (error) {
      console.error('Error loading model:', error);
    }
  }

  evaluate(testData) {
    let correct = 0;
    testData.forEach(([text, trueCategory]) => {
      const predictedCategory = this.classify(text);
      if (predictedCategory === trueCategory) {
        correct++;
      }
    });
    return correct / testData.length;
  }

  precisionRecallF1(testData) {
    const results = { TP: 0, FP: 0, FN: 0 };
    const categoryCounts = {};

    testData.forEach(([text, trueCategory]) => {
      const predictedCategory = this.classify(text);
      if (!categoryCounts[trueCategory]) {
        categoryCounts[trueCategory] = { TP: 0, FP: 0, FN: 0 };
      }
      if (predictedCategory === trueCategory) {
        categoryCounts[trueCategory].TP++;
      } else {
        categoryCounts[trueCategory].FN++;
        if (!categoryCounts[predictedCategory]) {
          categoryCounts[predictedCategory] = { TP: 0, FP: 0, FN: 0 };
        }
        categoryCounts[predictedCategory].FP++;
      }
    });

    Object.values(categoryCounts).forEach(counts => {
      results.TP += counts.TP;
      results.FP += counts.FP;
      results.FN += counts.FN;
    });

    const precision = results.TP / (results.TP + results.FP);
    const recall = results.TP / (results.TP + results.FN);
    const f1 = 2 * (precision * recall) / (precision + recall);

    return { precision, recall, f1 };
  }

  crossValidate(data, k = 5) {
    const foldSize = Math.floor(data.length / k);
    const accuracyResults = [];

    for (let i = 0; i < k; i++) {
      const validationStart = i * foldSize;
      const validationEnd = validationStart + foldSize;
      const validationSet = data.slice(validationStart, validationEnd);
      const trainingSet = [
        ...data.slice(0, validationStart),
        ...data.slice(validationEnd)
      ];

      const classifier = new NaiveBayesClassifier(this.stopWords, this.n);
      trainingSet.forEach(([text, category]) => classifier.train(text, category));

      const accuracy = classifier.evaluate(validationSet);
      accuracyResults.push(accuracy);
    }

    return accuracyResults.reduce((sum, acc) => sum + acc, 0) / accuracyResults.length;
  }

  visualizeImportantWords() {
    Object.keys(this.categories).forEach(category => {
      console.log(`Important words for category: ${category}`);
      this.categories[category].tfidf.tfidfs((i, measure, key) => {
        console.log(`${key}: ${measure}`);
      });
    });
  }
}

export default NaiveBayesClassifier;
