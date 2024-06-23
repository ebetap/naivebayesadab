import fs from 'fs';
import _ from 'lodash';
import natural from 'natural';
import { TfIdf } from 'natural';
import { createRequire } from 'module';
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

  preprocess(text) {
    const words = WordTokenizer.tokenize(text.toLowerCase())
      .filter(word => !this.stopWords.has(word) && /\w/.test(word))
      .map(word => WordNetLemmatizer.lemmatize(PorterStemmer.stem(word)));

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
    if (!this.categories.has(category)) {
      this.categories.set(category, { total: 0, wordCount: new Map(), tfidf: new TfIdf() });
    }

    this.categories.get(category).tfidf.addDocument(words.join(' '));

    words.forEach(word => {
      const categoryInfo = this.categories.get(category);
      categoryInfo.wordCount.set(word, (categoryInfo.wordCount.get(word) || 0) + 1);
      this.vocab.add(word);
      categoryInfo.total++;
    });
  }

  classify(text) {
    const words = this.preprocess(text);
    const scores = {};

    this.categories.forEach((categoryInfo, category) => {
      scores[category] = Math.log((categoryInfo.total + 1) / (this.totalExamples() + this.categories.size));

      words.forEach(word => {
        const wordCount = categoryInfo.wordCount.get(word) || 0;
        scores[category] += Math.log((wordCount + 1) / (categoryInfo.total + this.vocab.size));
      });
    });

    return _.maxBy(Object.keys(scores), category => scores[category]);
  }

  totalExamples() {
    return Array.from(this.categories.values()).reduce((total, category) => total + category.total, 0);
  }

  saveModel(filepath) {
    try {
      const modelData = {
        categories: Array.from(this.categories),
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
      this.categories = new Map(modelData.categories);
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
    const categoryCounts = new Map();

    testData.forEach(([text, trueCategory]) => {
      const predictedCategory = this.classify(text);
      if (!categoryCounts.has(trueCategory)) {
        categoryCounts.set(trueCategory, { TP: 0, FP: 0, FN: 0 });
      }
      if (predictedCategory === trueCategory) {
        categoryCounts.get(trueCategory).TP++;
      } else {
        categoryCounts.get(trueCategory).FN++;
        if (!categoryCounts.has(predictedCategory)) {
          categoryCounts.set(predictedCategory, { TP: 0, FP: 0, FN: 0 });
        }
        categoryCounts.get(predictedCategory).FP++;
      }
    });

    categoryCounts.forEach(counts => {
      results.TP += counts.TP;
      results.FP += counts.FP;
      results.FN += counts.FN;
    });

    const precision = results.TP / (results.TP + results.FP);
    const recall = results.TP / (results.TP + results.FN);
    const f1 = 2 * (precision * recall) / (precision + recall);

    return { precision, recall, f1 };
  }

  stratifyFolds(data, k) {
    const folds = Array.from({ length: k }, () => []);
    const categories = _.groupBy(data, ([, category]) => category);

    Object.values(categories).forEach(categoryData => {
      categoryData.forEach((item, index) => {
        folds[index % k].push(item);
      });
    });

    return folds;
  }

  crossValidate(data, k = 5) {
    const stratifiedFolds = this.stratifyFolds(data, k);
    const accuracyResults = [];

    for (let i = 0; i < k; i++) {
      const validationSet = stratifiedFolds[i];
      const trainingSet = stratifiedFolds
        .filter((_, index) => index !== i)
        .flat();

      const classifier = new NaiveBayesClassifier(this.stopWords, this.n);
      trainingSet.forEach(([text, category]) => classifier.train(text, category));

      const accuracy = classifier.evaluate(validationSet);
      accuracyResults.push(accuracy);
    }

    return accuracyResults.reduce((sum, acc) => sum + acc, 0) / accuracyResults.length;
  }

  visualizeImportantWords() {
    const d3n = new D3Node();
    const d3 = d3n.d3;

    this.categories.forEach((categoryInfo, category) => {
      console.log(`Important words for category: ${category}`);
      const words = [];
      categoryInfo.tfidf.tfidfs((i, measure, key) => {
        words.push({ word: key, score: measure });
      });

      // Sort words by TF-IDF score
      words.sort((a, b) => b.score - a.score);

      // Take top 10 words for visualization
      const topWords = words.slice(0, 10);

      // Create bar chart
      const svg = d3n.createSVG(600, 400);
      const margin = { top: 20, right: 20, bottom: 30, left: 40 };
      const width = +svg.attr('width') - margin.left - margin.right;
      const height = +svg.attr('height') - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleBand()
        .rangeRound([0, width])
        .padding(0.1)
        .domain(topWords.map(d => d.word));

      const y = d3.scaleLinear()
        .rangeRound([height, 0])
        .domain([0, d3.max(topWords, d => d.score)]);

      g.append('g')
        .attr('class', 'axis axis--x')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));

      g.append('g')
        .attr('class', 'axis axis--y')
        .call(d3.axisLeft(y).ticks(10));

      g.selectAll('.bar')
        .data(topWords)
        .enter().append('rect')
        .attr('class', 'bar')
        .attr('x', d => x(d.word))
        .attr('y', d => y(d.score))
        .attr('width', x.bandwidth())
        .attr('height', d => height - y(d.score));

      console.log(d3n.svgString());
    });
  }
}

// Utility function for parallel processing
function runInWorker(data, callback) {
  const worker = new Worker(__filename);
  worker.postMessage(data);
  worker.on('message', callback);
  worker.on('error', err => console.error('Worker error:', err));
}

if (!isMainThread) {
  parentPort.on('message', data => {
    const { trainingSet, text } = data;
    const classifier = new NaiveBayesClassifier();
    trainingSet.forEach(([text, category]) => classifier.train(text, category));
    const result = classifier.classify(text);
    parentPort.postMessage(result);
  });
}

export default NaiveBayesClassifier;
