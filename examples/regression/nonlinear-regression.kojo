// #include /nn.kojo
// #include /plot.kojo

cleari()
clearOutput()

val a = 2
val b = 3
val c = 1

val xData0 = Array.tabulate(20)(e => (e + 1).toDouble)
val yData0 = xData0 map (x => a * x * x + b * x + c + random(-5, 5))

val xNormalizer = new StandardScaler()
val yNormalizer = new StandardScaler()

val xData = xNormalizer.fitTransform(xData0)
val yData = yNormalizer.fitTransform(yData0)

val chart = scatterChart("Regression Data", "X", "Y", xData0, yData0)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val xDataf = xData.map(_.toFloat)
val yDataf = yData.map(_.toFloat)

Using.Manager { use =>
    val model = use(new NonlinearModel)
    model.train(xDataf, yDataf)
    val yPreds = model.predict(xDataf)
    val yPreds0 = yNormalizer.inverseTransform(yPreds.map(_.toDouble))
    addLineToChart(chart, Some("model"), xData0, yPreds0)
    updateChart(chart)
}.get

class NonlinearModel extends AutoCloseable {
    val LEARNING_RATE: Float = 0.1f
    val nd = NDManager.newBaseManager()
    nd.getManagedArrays.asScala.foreach { nda =>
        println(s"initial array with shape - ${nda.getShape}")
    }

    val hidden = 50

    val w1 = nd.randomNormal(0, 0.1f, new Shape(1, hidden), DataType.FLOAT32)
    val b1 = nd.zeros(new Shape(hidden))

    val w2 = nd.randomNormal(0, 0.1f, new Shape(hidden, 1), DataType.FLOAT32)
    val b2 = nd.zeros(new Shape(1))

    val params = new NDList(w1, b1, w2, b2).asScala

    params.foreach { p =>
        p.setRequiresGradient(true)
    }

    def newGradientCollector = {
        Engine.getInstance.newGradientCollector()
    }

    def modelFunction(x: NDArray): NDArray = {
        val l1 = x.matMul(w1).add(b1)
        val l1a = Activation.relu(l1)
        l1a.matMul(w2).add(b2)
    }

    def train(xValues: Array[Float], yValues: Array[Float]): Unit = {
        val x = nd.create(xValues).reshape(new Shape(-1, 1))
        val y = nd.create(yValues).reshape(new Shape(-1, 1))
        for (epoch <- 1 to 500) {
            Using.Manager { use =>
                val gc = use(newGradientCollector)
                //                    gc.zeroGradients()
                val yPred = use(modelFunction(x))
                val loss = use(y.sub(yPred).square().mean())
                if (epoch % 50 == 0) {
                    println(s"Loss -- ${loss.getFloat()}")
                }
                gc.backward(loss)
            }.get

            params.foreach { p =>
                p.subi(p.getGradient.mul(LEARNING_RATE))
                p.zeroGradients()
            }
        }
        println("Training Done")
    }

    def predict(xValues: Array[Float]): Array[Float] = {
        val x = nd.create(xValues).reshape(new Shape(-1, 1))
        val y = modelFunction(x)
        y.toFloatArray
    }

    def close() {
        println("Closing remaining ndarrays...")
        params.foreach(_.close())
        nd.close()
        println("Done")
    }
}
