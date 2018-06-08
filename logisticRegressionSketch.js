/*
 * Author: Michael A. Ortiz
 * Logistic Regression using Tensorflow.js
 * 
 * Inspired by the work of Daniel Shiffman's Coding Train
 * YouTube video here: https://www.youtube.com/watch?v=dLp10CFIvxI
 * in which Daniel demonstrates linear regression in Tensorflow.js
 * 
 * In this work, I extend Daniel's idea to present a an example of using
 * logistic regression for binary classification using Tensorflow.js.
 * 
 * The logic demonstrates the decision boundary which separates the red dots
 * from the blue dots.
 * 
 * At application start, 20 random points are generated.
 * The points are displayed as colored dots (10 blue dots and 10 red dots) and
 * are subsequently loaded into a logistic regression algorithm to determine
 * a decision boundary which best separates the red dots from the blue dots. 
 * The decision boundary is displayed as a black line onscreen.
 *  
 * The decision boundary is continually iterated on with each call to the draw().
 * 
 * The user may get predictions from the model by clicking or dragging (for multiple predictions)
 * the mouse on screen. 
 * 
 * The application will color the new dot based on the logistic regression model's prediction.
 * Concretely, the dot will be colored red if the new data point meets the threshold for 
 * the binary classification model. The dot will be colored blue otherwise.
 * 
 * The new data point is then added to the existing data and used in further iterations
 * on the decision boundary.
 * 
 * The weights and biases of the model as well as the current cost 
 * are indicated to the user onscreen in text form continually.
 */ 
let blue_dots = [[randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0],
                [randn_bm(), randn_bm(), 0]              
              ];

let red_dots = [[randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1],
                [randn_bm()-0.3, randn_bm(), 1]
              ];

let w = tf.variable(tf.randomUniform([1,2], 0, 1));
let b = tf.variable(tf.randomUniform([1,1], 0, 1)); 
const learningRate = 1;
const optimizer = tf.train.sgd(learningRate);
const threshold = 0.5;

// Normal distribution between 0 and 1 using Box-Muller transform.
// found here: https://stackoverflow.com/a/49434653
function randn_bm() {
  var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    num = num / 10.0 + 0.5; // Translate to 0 -> 1
    if (num > 1 || num < 0) return randn_bm(); // resample between 0 and 1
    return num;
}

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

function createPoint(mouseX, mouseY) {
  let point = transform(mouseX, mouseY);
  
  let prediction = tf.tidy(() => feedForward(tf.tensor2d(point, [1,2])));
  let predictionDataPromise = prediction.data();
  predictionDataPromise.then( value => {
    prediction.dispose();

    if (value < threshold) {
        blue_dots.push([point[0], point[1], 0]);
    } else {
        red_dots.push([point[0], point[1], 1]);
    }
  });
}

function mouseDragged() {
  createPoint(mouseX, mouseY);
}

function mousePressed() {
  createPoint(mouseX, mouseY);
}

function feedForward(x) {
  let z = x.matMul(w, false, true).add(b);
  return tf.sigmoid(z);
}

function loss(predictions, truth) {
  return tf.tidy(() => tf.losses.logLoss(truth, predictions));
}

function drawDots(dots) {
  for (let i = 0; i < dots.length; i++) {
    dot = inverseTransform(dots[i][0], dots[i][1]);
    point(dot[0], dot[1]);
  }
}

function draw() {
  background(255);
  
  stroke('blue');
  strokeWeight(20);
  drawDots(blue_dots);

  stroke('red');
  strokeWeight(20);
  drawDots(red_dots);

  let concat = blue_dots.concat(red_dots);
  let costMessage = "Cost: ";
  if (concat.length > 0) {
    let xs = concat.map(xy => xy.slice(0,2));
    let ys = concat.map(y => y[2]);
    let cost = optimizer.minimize(() => loss(feedForward(tf.tensor2d(xs, [xs.length, 2])),
                                                          tf.tensor2d(ys, [ys.length, 1])),
                                                          true);
    
    costMessage += cost.dataSync()[0];
    cost.dispose();
  }
  
  let weights = w.dataSync(); 
  let bias = b.dataSync();

  stroke(0);
  strokeWeight(5);
  line(inverseTransformX(0),
       inverseTransformY((-bias[0] -weights[0]*0)/(weights[1])),
       inverseTransformX(1),
       inverseTransformY((-bias[0] - weights[0]*1 )/ (weights[1])));

  textSize(40);
  stroke(0);
  strokeWeight(0);
  text("Bias: " + bias[0], 0, 682);
  text("Weight 1: " + weights[0], 0, 742);
  text("Weight 2: " + weights[1], 0, 802);
  text(costMessage, 0, 862);
  // console.log(tf.memory().numTensors);
}