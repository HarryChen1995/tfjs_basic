const tf = require("@tensorflow/tfjs")

var X = []

for (var i=0 ; i< 10; i++){
   X.push(i)
}
var Y = []

X.forEach((d)=>{Y.push(10*d+2)})
var Y_noise = []
Y.forEach((d)=>{Y_noise.push(d+Math.random())})
console.log(Y_noise)

X = tf.tensor(X)
Y = tf.tensor(Y)
const A = tf.scalar(Math.random()).variable();
const B = tf.scalar(Math.random()).variable();
A.print()
B.print()

const pred = (x) => tf.tidy(()=>A.mul(x).add(B))


const loss = (predict, label)=> tf.tidy(()=>predict.sub(label).square().mean())


const optimizer = tf.train.sgd(0.01)

for (var i = 0; i < 1000; i++){

    optimizer.minimize(()=>loss(pred(X), Y_noise))
}

console.log(
    `a: ${A.dataSync()}, B: ${B.dataSync()}`);
const preds = pred(X).dataSync();
preds.forEach((pred, i) => {
  console.log(`X: ${i}, pred: ${pred}`);
});








