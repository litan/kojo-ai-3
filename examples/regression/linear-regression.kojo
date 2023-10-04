// #include /nn.kojo
// #include /plot.kojo

cleari()
clearOutput()

val m = 10
val c = 3
val xData = Array.tabulate(20)(e => (e + 1.0))
val yData = xData map (x => x * m + c + randomDouble(-0.9, 0.9))

val chart = scatterChart("Regression Data", "X", "Y", xData, yData)
chart.getStyler.setLegendVisible(true)
drawChart(chart)

val xDataf = xData.map(_.toFloat)
val yDataf = yData.map(_.toFloat)

Using.Manager { use =>
    val model = use(new Model)
    model.train(xDataf, yDataf)
    val yPreds = model.predict(xDataf).map(_.toDouble)
    addLineToChart(chart, Some("model"), xData, yPreds)
    updateChart(chart)
}.get

class Model extends AutoCloseable {
    val LEARNING_RATE: Float = 0.005f
    val nd = NDManager.newBaseManager()

    val w = nd.create(1f).reshape(new Shape(1, 1))
    val b = nd.create(0f).reshape(new Shape(1))

    val params = new NDList(w, b).asScala

    params.foreach { p =>
        p.setRequiresGradient(true)
    }

    def newGradientCollector = {
        Engine.getInstance.newGradientCollector()
    }

    def train(xValues: Array[Float], yValues: Array[Float]): Unit = {
        val x = nd.create(xValues).reshape(new Shape(-1, 1))
        val y = nd.create(yValues).reshape(new Shape(-1, 1))
        for (epoch <- 1 to 10) {
            Using.Manager { use =>
                val gc = use(newGradientCollector)
                val yPred = use(x matMul w add b)
                val loss = use(yPred.sub(y).pow(2).mean())
                gc.backward(loss)
            }.get // force an exception if there was a problem

            params.foreach { p =>
                p.subi(p.getGradient.mul(LEARNING_RATE))
                p.zeroGradients()
            }
        }
        println("Training Done")
        println(w)
        println(b)
    }

    def predict(xValues: Array[Float]): Array[Float] = {
        val x = nd.create(xValues).reshape(new Shape(-1, 1))
        val y = x matMul w add b
        y.toFloatArray
    }

    def close() {
        println("Closing remaining ndarrays...")
        params.foreach { p =>
            p.close()
        }
        nd.close()
        println("Done")
    }
}
