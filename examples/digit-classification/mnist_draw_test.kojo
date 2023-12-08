// Copyright © Anay Kamat and Lalit Pant
// Draw digits and see predictions
// Draw in the whole red rectangular area for best results

// Note -- this is not the ideal model for computer vision as it uses
// only fully connected layers, and is thus not translation invariant
// You can see this limitation by drawing 1 on the sides

// #include /nn.kojo
cleari()
clearOutput()
disablePanAndZoom()
val cb = canvasBounds

import java.io.File
import java.awt.image.BufferedImage
import net.kogics.kojo.util.Utils

// neural network stuff

class MnistModel(modelFile: String) {
    val nd = NDManager.newBaseManager()

    val params = load(6)
    val w1 = params(0); val b1 = params(1)
    val w2 = params(2); val b2 = params(3)
    val w3 = params(4); val b3 = params(5)

    def load(n: Int): Seq[NDArray] = {
        import java.io._
        println(s"Loading model from file - $modelFile")
        Using.Manager { use =>
            val dis = use(new DataInputStream(
                new BufferedInputStream(new FileInputStream(modelFile))
            ))
            val header = dis.readChar()
            assert(header == 'P', "Not a valid model file - unknown param header")
            val params = ArrayBuffer.empty[NDArray]
            for (i <- 1 to n) {
                val p = nd.decode(dis)
                params.append(p)
            }
            params.toArray
        }.get
    }

    def modelFunction(x: NDArray): NDArray = {
        val l1 = x.matMul(w1).add(b1)
        val l1a = Activation.relu(l1)

        val l2 = l1a.matMul(w2).add(b2)
        val l2a = Activation.relu(l2)

        l2a.matMul(w3).add(b3).softmax(1)
    }

    def predict(x: NDArray): Long = {
        modelFunction(x).argMax(1).getLong()
    }

    def predictProbs(x: NDArray): Array[Float] = {
        modelFunction(x).toFloatArray
    }

    def close() {
        println("Closing ndarrays...")
        params.foreach(_.close())
        nd.close()
        println("Done")
    }
}

def imgBlueToNDArray(image: BufferedImage, nd: NDManager): NDArray = {
    import java.nio.FloatBuffer
    val h = image.getHeight
    val w = image.getWidth
    val imgBuffer = FloatBuffer.allocate(h * w * 1 * 4)

    for (y <- 0 until h) {
        for (x <- 0 until w) {
            val pixel = image.getRGB(x, y)
            val blue = pixel & 0xFF
            imgBuffer.put(blue.toFloat / 255f)
        }
    }
    imgBuffer.flip()
    val shape = Shape(1, image.getHeight * image.getWidth)
    val t2 = nd.create(imgBuffer, shape)
    t2
}

def imgBlueToGray(image: BufferedImage): BufferedImage = {
    val h = image.getHeight
    val w = image.getWidth
    val newImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)

    for (y <- 0 until h) {
        for (x <- 0 until w) {
            val pixel = image.getRGB(x, y)
            val blue = pixel & 0xFF
            newImage.setRGB(x, y, blue << 16 | blue << 8 | blue)
        }
    }
    newImage
}

val scriptDir = Utils.kojoCtx.baseDir
val modelFile = s"${kojoCtx.baseDir}/mnist.djl.model"

require(
    new File(modelFile).exists,
    s"Cannot find MNIST model file: ${modelFile}")

val mnistNet = new MnistModel(modelFile)

def predict(inputTensor: NDArray): Int = {
    mnistNet.predict(inputTensor).toInt
}

def predictProbs(inputTensor: NDArray): Seq[(Int, String)] = {
    val probs = mnistNet.predictProbs(inputTensor)
    for ((prob, idx) <- probs.zipWithIndex) yield {
        (idx, f"${prob}%.2f")
    }
}

// drawing stuff

def resultPic(digitImg: BufferedImage, probs: Array[Float]): Picture = {
    val netInput = picRowCentered(
        Picture.text("Net Input: ").withPenColor(cm.darkBlue),
        Picture.hgap(5),
        Picture.image(digitImg)
    )
    val pics = ArrayBuffer[Picture](netInput, Picture.vgap(10))
    probs.zipWithIndex.foreach { case (p, idx) =>
        val pic = picRowCentered(
            Picture.text(idx).withPenColor(black),
            Picture.hgap(10),
            picStack(
                Picture.rectangle(100, 20).withNoPen(),
                Picture.rectangle(100 * p, 20).withFillColor(darkGray).withNoPen()
            ),
            Picture.hgap(10),
            Picture.text(f"${p}%.2f").withPenColor(black),
        )
        pics.append(pic)
        pics.append(Picture.vgap(5))
    }
    pics.append(Picture.vgap(10))
    pics.append(Picture.text("Prediction Probabilities", 20).withPenColor(cm.darkBlue))
    picColCentered(pics)
}

def pictureFromPath(path: List[(Double, Double)]): Picture = {
    Picture.fromVertexShape { shape =>
        shape.beginShape()
        for ((x, y) <- path) {
            shape.vertex(x, y)
        }
        shape.endShape()
    }
}

case class State(
    paths:     List[(Double, Double)] = List(),
    isDrawing: Boolean                = false
)

def initialState: State = State()

def go() {
    animateWithState(initialState) { state =>
        erasePictures()

        val bgPic = Picture.rect(100, 100).thatsTranslated(-50, -50)
        bgPic.draw()

        val pic = picStack(
            Picture.rect(100, 100).thatsTranslated(-50, -50).withPenColor(white).withNoPen,
            if (!state.paths.isEmpty)
                pictureFromPath(state.paths).withPenColor(blue).withPenThickness(14)
            else
                Picture {}
        )

        pic.draw()

        if (state.isDrawing && !isMousePressed) {
            ndScoped { _ =>
                val digitLargeImg = pic.toImage
                val digitImg = new java.awt.image.BufferedImage(28, 28, java.awt.image.BufferedImage.TYPE_INT_RGB)
                val g = digitImg.createGraphics
                g.translate(0, 28)
                g.scale(1, -1)
                g.drawImage(digitLargeImg, 3, 3, 22, 22, null)
                g.dispose()

                val digitNDArray = imgBlueToNDArray(digitImg, mnistNet.nd)
                val prediction = predict(digitNDArray)
                val predictionProbs = mnistNet.predictProbs(digitNDArray)

                stopAnimation()
                val digitImgActual = imgBlueToGray(digitImg)
                draw(resultPic(digitImgActual, predictionProbs).withTranslation(200, -150))
            }
        }

        draw(Picture.widget(clearBtn).withPosition(-36, -150))

        (state.isDrawing, isMousePressed) match {
            case (false, false) => state
            case (false, true) => {
                val mouseData = (mouseX, mouseY)
                State(
                    paths = List(mouseData),
                    isDrawing = true
                )
            }
            case (true, false) => state.copy(isDrawing = false)
            case (true, true) => {
                val mouseData = (mouseX, mouseY)
                state.copy(
                    paths = state.paths.appended(mouseData),
                    isDrawing = true
                )
            }
        }
    }
}

val clearBtn = Button("Next") {
    go()
}

go()
