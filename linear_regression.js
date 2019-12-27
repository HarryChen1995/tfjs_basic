const tf = require("@tensorflow/tfjs")
var X = []

for (var i=0 ; i< 10; i++){
   X.push(i)
}
var Y = []

X.forEach((d)=>{Y.push(10*d+2)})
var Y_noise = []
Y.forEach((d)=> {Y_noise.push(d+Math.random())})

X = tf.tensor(X)
Y = tf.tensor(Y)

const A = tf.scalar(Math.random()).variable();
const B = tf.scalar(Math.random()).variable();

const pred = (x) => A.mul(x).add(B)


const loss = (predict, label)=> predict.sub(label).square().mean()


const optimizer = tf.train.sgd(0.01)

for (var i = 0; i < 1000; i++){

    optimizer.minimize(()=>loss(pred(X), Y_noise))
}

var s  = `A: ${A.dataSync()}, B: ${B.dataSync()}\n`;
const preds = pred(X).dataSync();
preds.forEach((pred, i) => {
  s += `X: ${i}, pred: ${pred}` + "\n";
});


console.log(s)





