// #include /nn.kojo

import java.io.File
import java.awt.image.BufferedImage

import net.kogics.kojo.util.Utils

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

    def close() {
        println("Closing ndarrays...")
        params.foreach(_.close())
        nd.close()
        println("Done")
    }
}

def imgGrayToNDArray(image: BufferedImage, nd: NDManager): NDArray = {
    import java.nio.FloatBuffer
    val h = image.getHeight
    val w = image.getWidth
    val imgBuffer = FloatBuffer.allocate(h * w * 1 * 4)

    for (y <- 0 until h) {
        for (x <- 0 until w) {
            val pixel = image.getRGB(x, y)
            val gray = (pixel >> 16) & 0xff
            imgBuffer.put(gray.toFloat / 255f)
        }
    }
    imgBuffer.flip()
    val shape = Shape(1, image.getHeight * image.getWidth)
    val t2 = nd.create(imgBuffer, shape)
    t2
}

cleari()

val scriptDir = Utils.kojoCtx.baseDir
val modelFile = s"${kojoCtx.baseDir}/mnist.djl.model"

require(
    new File(modelFile).exists,
    s"Cannot find MNIST model file: ${modelFile}")

val mnistNet = new MnistModel(modelFile)

def predict(inputTensor: NDArray): Int = {
    mnistNet.predict(inputTensor).toInt
}

def makePredictionPic(filename: String, expected: Int): Picture = {
    ndScoped { _ =>
        val inputImage = Utils.loadBufImage(filename)
        val pic = Picture.image(inputImage)
        val inputTensor = imgGrayToNDArray(inputImage, mnistNet.nd)
        val prediction = predict(inputTensor)
        val clr = if (prediction == expected) green else red
        val pic2 = Picture.text(prediction, 30).withPenColor(clr)
        picRowCentered(pic, Picture.hgap(20), pic2)
    }
}

val pics = for (i <- 0 to 9) yield {
    makePredictionPic(s"test/$i.png", i)
}

val pics2 = picCol(pics)
drawCentered(pics2)

mnistNet.close()

//cleari()
//val pic = makePredictionPic("test/3.png", 1)
//drawCentered(pic)
