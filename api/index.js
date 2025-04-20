const express = require("express");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const axios = require("axios");

const app = express();
app.use(bodyParser.json());

const MODEL_URL = "https://github.com/alaa-sanad/halalapi/releases/download/v1/model.json";
const TOKENIZER_URL = "https://github.com/alaa-sanad/halalapi/releases/download/v1/tokenizer.json";
const MODEL_DIR = "/tmp/model";
const TOKENIZER_PATH = path.join("/tmp", "tokenizer.json");

let model;
let tokenizer;
const MAX_LENGTH = 50;

async function downloadFile(url, outputPath) {
  const writer = fs.createWriteStream(outputPath);
  const response = await axios({
    method: "GET",
    url,
    responseType: "stream"
  });
  response.data.pipe(writer);
  return new Promise((resolve, reject) => {
    writer.on("finish", resolve);
    writer.on("error", reject);
  });
}

async function loadTokenizer() {
  if (!fs.existsSync(TOKENIZER_PATH)) {
    console.log("📥 Downloading tokenizer...");
    await downloadFile(TOKENIZER_URL, TOKENIZER_PATH);
    console.log("✅ Tokenizer downloaded.");
  }
  const tokenizerData = fs.readFileSync(TOKENIZER_PATH, "utf-8");
  tokenizer = JSON.parse(tokenizerData);
  console.log("✅ Tokenizer loaded!");
}

async function loadModel() {
  if (!model) {
    if (!fs.existsSync(MODEL_DIR)) fs.mkdirSync(MODEL_DIR);

    const modelPath = "file://" + path.join(MODEL_DIR, "model.json");
    const modelJsonPath = path.join(MODEL_DIR, "model.json");

    if (!fs.existsSync(modelJsonPath)) {
      console.log("📥 Downloading model...");
      await downloadFile(MODEL_URL, modelJsonPath);
      // TensorFlow.js will also look for `group1-shard1of1.bin` in the same folder
      const binUrl = MODEL_URL.replace("model.json", "group1-shard1of1.bin");
      const binPath = path.join(MODEL_DIR, "group1-shard1of1.bin");
      await downloadFile(binUrl, binPath);
      console.log("✅ Model files downloaded.");
    }

    model = await tf.loadLayersModel(modelPath);
    console.log("✅ Model loaded!");
  }
  return model;
}

function translateText(text) {
  return text.toLowerCase();
}

app.post("/api/predict", async (req, res) => {
  try {
    const ingredients = req.body.ingredients || [];
    if (!ingredients.length) {
      return res.status(400).json({ error: "❌ No ingredients provided!" });
    }

    await loadTokenizer();
    await loadModel();

    const translated = ingredients.map(translateText);
    const sequences = translated.map(text =>
      text.split(" ").map(word => tokenizer.word_index[word] || tokenizer.word_index["<OOV>"] || 0)
    );

    const padded = sequences.map(seq => {
      const padding = new Array(MAX_LENGTH - seq.length).fill(0);
      return seq.length >= MAX_LENGTH
        ? seq.slice(0, MAX_LENGTH)
        : seq.concat(padding);
    });

    const inputTensor = tf.tensor2d(padded);
    const predictions = model.predict(inputTensor).dataSync();

    const results = ingredients.map((ing, i) => {
      const score = predictions[i];
      let classification = "doubtful";
      if (score >= 0.7) classification = "halal";
      else if (score <= 0.3) classification = "haram";
      return { ingredient: ing, classification };
    });

    res.status(200).json({
      ingredients: results,
      overall_classification: results.some(r => r.classification === "haram")
        ? "haram"
        : results.some(r => r.classification === "doubtful")
        ? "doubtful"
        : "halal"
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

module.exports = app;
