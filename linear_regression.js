const tf = require("@tensorflow/tfjs-node")
var X = []

for (var i=0 ; i< 50; i++){
   X.push(i)
}
var Y = []

X.forEach((d)=>{Y.push(10*d+2)})
var Y_noise = []
Y.forEach((d)=> {Y_noise.push(d+Math.random())})
Y_True = Y
X = tf.tensor(X, [1, 50], "float32")
Y = tf.tensor(Y, [1, 50], "float32")

const A = tf.variable(tf.randomNormal([1]))
const B = tf.variable(tf.randomNormal([1]));
A_orignal = A.dataSync()
B_orignal = B.dataSync()
const pred = (x) => A.mul(x).add(B)


const loss = (predict, label)=> predict.sub(label).square().mean()


const optimizer = tf.train.sgd(0.001)

train = function(){
  optimizer.minimize(()=>loss(pred(X), Y_noise))
}
for (var i = 0; i < 5000; i++){

    tf.tidy(train)
   
}
var s= `A Before Training : ${A_orignal}, B Before Training: ${B_orignal}\n`
 s  += `A After Training: ${A.dataSync()}, B After Training: ${B.dataSync()}\n`;
const preds = pred(X).dataSync();
preds.forEach((pred, i) => {
  s += `X: ${i}, pred: ${pred}, true:${Y_True[i]}` + "\n";
});


console.log(s)





