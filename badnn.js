

// takes in weights, which is array of arrays of numbers
// these nets are rectangular only, meaning that you have some amount of inputs -> rectangle of hidden nodes -> some amount of outputs

// params format:
// p[0]: list of biases of inputs -> w[0].length == numInputs
// p[1]: list of weights between inputs, hidden layer 1 -> w[1].length == numInputs * layerHeight
// p[2]: list of biases of hidden layer 1 -> w[2].length == layerHeight
// p[3]: list of weights between layer 1, layer 2 -> w[2].length == layerHeight * layerHeight
// ...
// p[n-1]: list of weights between last layer, output -> w[n-1].length == layerHeight * numOutputs
// p[n]: list of biases of outputs ->  w[n].length == numOutputs

// for a list of weights:
// input nodes: n1, n2, ...
// output: m1, m2, ...
// list: [n1-m1, n1-m2, n1-..., n2-m1, n2-m2, n2-..., ...]

function NeuralNet(nin = 4, nout = 4, params = NeuralNet.prototype.createWeights(1, 2, 1, 1)){
    this.numInputs = nin
    this.numOutputs = nout
    this.p = params

    // weight dimensions checking
    this.verifyParams(this.p) // will throw error if not valid
}


NeuralNet.prototype.evaluate = function(input){

    // console.log(input, this.p)
    // input is an array of numbers of length numInputs

    let layerHeight = this.p.length > 3 ? this.p[2].length : this.numOutputs


    if(input.length != this.numInputs){
        throw new Error("NN: Cannot evaluate this input")
    }

    let inValues = input
    let outValues = new Array(layerHeight)
    outValues.fill(0)

    // console.log(inValues)

    // add input biases to working values
    for(var i = 0; i < inValues.length; i ++){
        inValues[i] += this.p[0][i]
    }
    // console.log(inValues, outValues) // checking references etc issues

    for(var i = 0; i < inValues.length; i ++){
        for(var ii = 0; ii < layerHeight; ii ++){
            // console.log(outValues[ii])
            outValues[ii] += this.p[1][i*layerHeight + ii]*inValues[i]
        }
    }

    inValues = outValues
    for(var i = 0; i < outValues.length; i ++){
        inValues[i] = this.activation(outValues[i])
    }

    for(var i = 2; i < this.p.length; i += 2){


        let nextLayerHeight = this.p.length - i > 3 ? layerHeight : this.numOutputs
        // console.log (i, 'calc type: ', this.p.length - i > 3 ? 'normal hidden' : (this.p.length - i > 1 ? 'final hidden' : 'outputs'))

        outValues = new Array(nextLayerHeight)
        outValues.fill(0)

        // console.log('curvalues ',inValues)

        if(this.p.length - i > 1){

            // add biases
            for(var ii = 0; ii < this.p[i].length; ii ++){
                inValues[ii]+=this.p[i][ii]
            }

            // apply weights
            for(var ii = 0; ii < inValues.length; ii ++){
                for(var iii = 0; iii < nextLayerHeight; iii ++){
                    // console.log('apply weight', ii*nextLayerHeight + iii, 'to next value', iii)
                    outValues[iii] += this.p[i+1][ii*nextLayerHeight + iii]*inValues[ii]
                }
            }

        }else{
            // add biases
            for(var ii = 0; ii < this.p[i].length; ii ++){
                outValues[ii]=this.p[i][ii] + inValues[ii]
            }
        }

        inValues = outValues
        for(var ii = 0; ii < outValues.length; ii ++){
            inValues[ii] = this.activation(outValues[ii])
        }
    }

    return outValues
}

NeuralNet.prototype.verifyParams = function(p){
    // verifies them in context of this nn's inputs, outputs

    // let e = new Error("NN: Invalid weights")

    // console.log(w)

    if(p.length < 3){throw e}
    if(p.length % 2 != 1){throw e}
    let layerHeight = this.numOutputs
    if(p.length > 3){
        layerHeight = p[2].length
    }

    for(var i = 0; i < p.length; i ++){
        if(i == 0){
            if(p[i].length != this.numInputs){throw new Error("Invalid Weights")}
            continue
        }
        if(i == 1){
            if(p[i].length != this.numInputs * layerHeight){throw new Error("Invalid Weights")}
            continue
        }
        if(i == p.length - 2){
            if(p[i].length != this.numOutputs * layerHeight){throw new Error("Invalid Weights")}
            continue
        }
        if(i == p.length - 1){
            if(p[i].length != this.numOutputs){throw new Error("Invalid Weights")}
            continue
        }
        if(i % 2 == 0){
            if(p[i].length != layerHeight){throw new Error("Invalid Weights")}
        }else{
            if(p[i].length != layerHeight * layerHeight){throw new Error("Invalid Weights")}
        }
    }
}

NeuralNet.prototype.createWeights = function(numLayers, layerHeight, numInputs = 4, numOutputs = 4, variation=1){
    let output = []

    // input biases
    let curLayer = []
    for(var i = 0; i < numInputs || this.numInputs; i ++){curLayer.push(this.generateWeightValue(variation))}
    output.push(curLayer)

    for(var i = 0; i < numLayers; i ++){
        // weights between previous, current layer
        curLayer = []
        let x = i > 0 ? layerHeight : (numInputs || this.numInputs)

        for(var ii = 0; ii < layerHeight * x; ii ++){
            curLayer.push(this.generateWeightValue(variation))
        }
        output.push(curLayer)

        // biases
        curLayer = []
        for(var ii = 0; ii < layerHeight; ii ++){
            curLayer.push(this.generateWeightValue(variation))
        }
        output.push(curLayer)
    }

    // weights between previous, outputs
    curLayer = []
    let x = numLayers > 0 ? layerHeight : (numOutputs || this.numOutputs)

    for(var i = 0; i < (numOutputs || this.numOutputs) * x; i ++){
        curLayer.push(this.generateWeightValue(variation))
    }
    output.push(curLayer)

    // biases for outputs
    curLayer = []
    for(var i = 0; i < numOutputs || this.numOutputs; i ++){curLayer.push(this.generateWeightValue(variation))}
    output.push(curLayer)

    return output
}


NeuralNet.prototype.generateWeightValue = function(variation = 1){
    // return Math.trunc(10*(Math.random()-0.5)*variation)/10
    return (Math.random() - 0.5) * variation
    // return Math.random() > 0.5 ? 1 : 0
    // return 1
}

NeuralNet.prototype.activation = function(x){
    // return Math.max(0, x)
    return x > 0 ? x : x*0.5
    // return x
}




NeuralNet.prototype.trainGradientDescent = function(fitnessFunc, iterations = 300){

    // "Gradient Descent"

    // todo implement a training timeout as well

    // fitnessFunc is a function taking a neural network object (this) as an argument
    // it returns a single number, which is fitness
    // ie, to train a number doubler:
    /*
        fitnessFunc = (nn)=>{
            let input = Math.random()*100
            let output = nn.evaluate([input])[0]

            return -Math.abs(input*2 - output)
        }
    */

    let iterationCount = 0
    while(iterationCount < iterations){

        if(iterationCount % 100 == 0){console.log(iterationCount, fitnessFunc(this))}

        let increment = Math.random()*(iterationCount < 200 ? 0.5 : 0.05)

        iterationCount++

        let gradient = [] // values in this will be set according to fitness
        for(var i = 0; i < this.p.length; i ++){
            gradient.push(new Array(this.p[i].length))
        }

        // for every parameter
        for(var i = 0; i < this.p.length; i ++){
            for(var ii = 0; ii < this.p[i].length; ii ++){
                this.p[i][ii] += increment
                let fitnessUp = fitnessFunc(this)
                this.p[i][ii] -= increment*2
                let fitnessDown = fitnessFunc(this)
                this.p[i][ii] += increment
                if(fitnessUp > fitnessDown){gradient[i][ii] = 1}else{gradient[i][ii] = -1}
            }
        }

        // apply gradient
        for(var i = 0; i < gradient.length; i ++){
            for(var ii = 0; ii < gradient[i].length; ii ++){

                this.p[i][ii] += increment * gradient[i][ii]

            }
        }

    }
    
}