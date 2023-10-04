// #include /nn.kojo
// #include /plot.kojo

import ai.djl.basicdataset.cv.classification.Mnist

cleari()
clearOutput()

Using.Manager { use =>
    val model = use(new MnistModel())
    timeit("Training") {
        model.train()
    }
    model.test()
    model.save()
}.get

def dataset(usage: Dataset.Usage, mgr: NDManager) = {
    val mnist =
        Mnist.builder()
            .optUsage(usage)
            .setSampling(64, true)
            .optManager(mgr)
            .build()

    mnist.prepare(new ProgressBar())
    mnist
}

class MnistModel extends AutoCloseable {
    def learningRate(e: Int) = e match {
        case n if n <= 10 => 0.1f
        case n if n <= 15 => 0.03f
        case _            => 0.01f
    }

    val nd = NDManager.newBaseManager()
    val trainingSet = dataset(Dataset.Usage.TRAIN, nd)
    val validateSet = dataset(Dataset.Usage.TEST, nd)

    val hidden1 = 38
    val hidden2 = 12

    val w1 = nd.randomNormal(0, 0.1f, new Shape(784, hidden1), DataType.FLOAT32)
    val b1 = nd.zeros(new Shape(hidden1))

    val w2 = nd.randomNormal(0, 0.1f, new Shape(hidden1, hidden2), DataType.FLOAT32)
    val b2 = nd.zeros(new Shape(hidden2))

    val w3 = nd.randomNormal(0, 0.1f, new Shape(hidden2, 10), DataType.FLOAT32)
    val b3 = nd.zeros(new Shape(10))

    val params = new NDList(w1, b1, w2, b2, w3, b3).asScala

    val softmax = new SoftmaxCrossEntropyLoss()

    params.foreach { p =>
        p.setRequiresGradient(true)
    }

    def newGradientCollector = {
        Engine.getInstance.newGradientCollector()
    }

    def modelFunction(x: NDArray): NDArray = {
        val l1 = x.matMul(w1).add(b1)
        val l1a = Activation.relu(l1)

        val l2 = l1a.matMul(w2).add(b2)
        val l2a = Activation.relu(l2)

        l2a.matMul(w3).add(b3)
    }

    val numEpochs = 20

    def train(): Unit = {
        val lossChart = new LiveChart(
            "Loss Plot", "epoch", "loss", 0, numEpochs, 0, 0.6
        )
        for (epoch <- 1 to numEpochs) {
            var eloss = 0f
            trainingSet.getData(nd).asScala.foreach { batch0 =>
                Using.Manager { use =>
                    val batch = use(batch0)
                    val gc = use(newGradientCollector)
                    //                        gc.zeroGradients()
                    val x = batch.getData.head.reshape(new Shape(-1, 784))
                    val y = batch.getLabels.head
                    val yPred = use(modelFunction(x))
                    val loss = use(softmax.evaluate(new NDList(y), new NDList(yPred)))
                    eloss = loss.getFloat()
                    gc.backward(loss)
                }.get

                params.foreach { p =>
                    p.subi(p.getGradient.mul(learningRate(epoch)))
                    p.zeroGradients()
                }
            }
            println(s"[$epoch] Loss -- $eloss")
            lossChart.update(epoch, eloss)
        }
        println("Training Done")
    }

    def test() {
        println("Determining accuracy on the test set")
        var total = 0l
        var totalGood = 0l
        validateSet.getData(nd).asScala.foreach { batch0 =>
            Using.Manager { use =>
                val batch = use(batch0)
                val x = batch.getData.head.reshape(new Shape(-1, 784))
                val y = batch.getLabels.head
                val yPred = use(modelFunction(x).softmax(1).argMax(1))
                val matches = use(y.toType(DataType.INT64, false).eq(yPred))
                total += matches.getShape.get(0)
                totalGood += matches.countNonzero.getLong()
            }.get
        }
        val acc = 1f * totalGood / total
        println(acc)
    }

    def save() {
        import java.io._
        val modelFile = s"${kojoCtx.baseDir}/mnist.djl.model"
        println(s"Saving model in file - $modelFile")
        Using.Manager { use =>
            val dos = use(new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(modelFile))
            ))
            dos.writeChar('P')
            params.foreach { p =>
                dos.write(p.encode())
            }
        }.get

    }

    def close() {
        println("Closing remaining ndarrays...")
        params.foreach(_.close())
        nd.close()
        println("Done")
    }
}
