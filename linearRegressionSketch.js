/*
 * Author: Michael A. Ortiz
 * 
 * Linear Regression with TensorFlow.js
 * 
 * A derivative of the work by Daniel Shiffman in the Coding Challenge YouTube video on the
 * CodingTrain channel found here: https://www.youtube.com/watch?v=dLp10CFIvxI
 *  
 * You can also find Daniel's original work here: 
 * https://github.com/CodingTrain/website/blob/master/CodingChallenges/CC_104_tf_linear_regression/sketch.js 
 */ 

let x_vals = [];
let y_vals = [];

let w = tf.variable(tf.scalar(Math.random()), trainable=true);
let b = tf.variable(tf.scalar(Math.random()), trainable=true); 
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);


function setup() {
  createCanvas(2400,2400);
}

function transformX(x) {
  let transformedX = map(x, 0, width, 0, 1);
  
  return transformedX;
}

function transformY(y) {
  let transfomredY = map(y, 0, height, 1, 0);

  return transfomredY;
}


function inverseTransformX(x) {
  let inverseTransformedX = map(x, 0, 1, 0, width);

  return inverseTransformedX;
}


function inverseTransformY(y) {
  let inverseTransformedY = map(y, 0, 1, height, 0);

  return inverseTransformedY;
}


function transform(x, y) {
  let transformedX = transformX(x);
  let transformedY = transformY(y);

  return [transformedX, transformedY];
}

function inverseTransform(x, y) {
  let inverseX = inverseTransformX(x);
  let inverseY = inverseTransformY(y);

  return [inverseX, inverseY];
}

function mousePressed() {
  createPoint(mouseX, mouseY);
}

function createPoint(mouseX, mouseY) {
  let point = transform(mouseX, mouseY);
  x_vals.push(point[0]);
  y_vals.push(point[1]);
}

function mouseDragged() {
  createPoint(mouseX, mouseY);
}

function feedForward(x) {
  return tf.tidy(() => tf.tensor1d(x).mul(w).add(b));
}

function loss(predictions, truth) {
  return tf.tidy(() => predictions.sub(tf.tensor1d(truth)).square().mean());
}

function draw() {
  background(255);
  
  
  stroke('red');
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    
    point(inverseTransformX(x_vals[i]), inverseTransformY(y_vals[i]));
  }

  let costMessage = "Cost: ";
  if (x_vals.length > 0) {
    let cost = optimizer.minimize(() => loss(feedForward(x_vals), y_vals), true);
    costMessage += cost.dataSync()[0];

    const regressionLineYs = feedForward([0,1]);
    ys = regressionLineYs.dataSync();
    regressionLineYs.dispose();

    stroke('black');
    strokeWeight(5);
    line(inverseTransformX(0), inverseTransformY(ys[0]), inverseTransformX(1), inverseTransformY(ys[1]));
  }

  stroke('black');
  let weight = w.dataSync();
  let bias = b.dataSync();
  textSize(60);
  strokeWeight(0);
  
  text("Bias: " + bias[0], 1040, 742);
  text("Weight: " + weight[0], 1040, 802);
  text(costMessage, 1040, 862);
  // console.log(tf.memory().numTensors);
}