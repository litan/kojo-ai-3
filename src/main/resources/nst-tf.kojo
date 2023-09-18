import java.awt.image.BufferedImage
import java.nio.file.Paths
import scala.util.Using
import ai.djl.repository.zoo.Criteria
import net.kogics.kojo.nst._

class NeuralStyleFilter(savedModelFile: String, styleImageFile: String, alpha: Float) extends ImageOp {
    val styleImage = image(styleImageFile)
    val criteria =
        Criteria.builder()
            .setTypes(classOf[NstInputData], classOf[BufferedImage])
            .optEngine("TensorFlow")
            .optTranslator(new TfNstTranslator())
            .optModelPath(Paths.get(savedModelFile))
            .optModelName("saved_model")
            .build()

    val mdl = criteria.loadModel()

    def filter(src: BufferedImage) = {
        Using.Manager { use =>
            val predictor = use(mdl.newPredictor())
            predictor.predict(NstInputData(src, styleImage, alpha))
        }.get
    }
}
