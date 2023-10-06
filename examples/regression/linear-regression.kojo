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

ndScoped { use =>
    val model = use(new Model)
    model.train(xDataf, yDataf)
    val yPreds = model.predict(xDataf).map(_.toDouble)
    addLineToChart(chart, Some("model"), xData, yPreds)
    updateChart(chart)
}

class Model extends AutoCloseable {
    val LEARNING_RATE: Float = 0.005f
    val nm = ndMaker

    val w = nm.create(1f).reshape(shape(1, 1))
    val b = nm.create(0f).reshape(shape(1))

    val params = new NDList(w, b).asScala

    params.foreach { p =>
        p.setRequiresGradient(true)
    }

    def train(xValues: Array[Float], yValues: Array[Float]): Unit = {
        val x = nm.create(xValues).reshape(shape(-1, 1))
        val y = nm.create(yValues).reshape(shape(-1, 1))
        for (epoch <- 1 to 10) {
            ndScoped { use =>
                val gc = use(gradientCollector)
                val yPred = x matMul w add b
                val loss = yPred.sub(y).pow(2).mean()
                gc.backward(loss)
            }

            ndScoped { _ =>
                params.foreach { p =>
                    p.subi(p.getGradient.mul(LEARNING_RATE))
                    p.zeroGradients()
                }
            }
        }
        x.close(); y.close()
        println("Training Done")
        println(w)
        println(b)
    }

    def predict(xValues: Array[Float]): Array[Float] = {
        ndScoped { _ =>
            val x = nm.create(xValues).reshape(shape(-1, 1))
            val y = x matMul w add b
            y.toFloatArray
        }
    }

    def close() {
        println("Closing remaining ndarrays...")
        params.foreach { p =>
            p.close()
        }
        nm.close()
        println("Done")
    }
}
