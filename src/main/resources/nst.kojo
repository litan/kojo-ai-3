import java.awt.image.BufferedImage
import scala.util.Using
import net.kogics.kojo.nst._

class NeuralStyleFilter(modelDir: String, styleImageFile: String, alpha: Float) extends ImageOp {
    val styleImage = image(styleImageFile)
    val mdl = cachedModel(modelDir, "PyTorch")

    def filter(src: BufferedImage) = {
        Using.Manager { use =>
            val predictor = use(mdl.newPredictor())
            predictor.predict(NstInputData(src, styleImage, alpha))
        }.get
    }
}
