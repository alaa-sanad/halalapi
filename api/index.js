
const express = require("express");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(bodyParser.json());

let model;
let tokenizer;
const MAX_LENGTH = 50;

async function loadModel() {
  model = await tf.loadLayersModel("file://model/model.json");
  console.log("✅ Model loaded!");
}

function loadTokenizer() {
  const tokenizerData = fs.readFileSync("tokenizer.json", "utf-8");
  tokenizer = JSON.parse(tokenizerData);
  console.log("✅ Tokenizer loaded!");
}

loadModel();
loadTokenizer();

function translateText(text) {
  return text.toLowerCase();
}

app.post("/api/predict", async (req, res) => {
  try {
    const ingredients = req.body.ingredients || [];
    if (!ingredients.length) {
      return res.status(400).json({ error: "❌ No ingredients provided!" });
    }

    const translated = ingredients.map(translateText);
    const sequences = translated.map(text =>
      (text.split(" ").map(word => tokenizer.word_index[word] || tokenizer.word_index["<OOV>"] || 0))
    );

    const padded = sequences.map(seq => {
      const padding = new Array(MAX_LENGTH - seq.length).fill(0);
      return (seq.length >= MAX_LENGTH)
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
