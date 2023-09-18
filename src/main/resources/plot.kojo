import net.kogics.kojo.plot._

import org.knowm.xchart.internal.chartpart.Chart
import org.knowm.xchart.internal.series.Series
import org.knowm.xchart.style.Styler

def drawChart[A <: Styler, B <: Series](chart: Chart[A, B]) {
    import org.knowm.xchart.BitmapEncoder
    cleari()
    val cb = canvasBounds
    val img = BitmapEncoder.getBufferedImage(chart)
    val pic = trans(
        cb.x + (cb.width - img.getWidth) / 2,
        cb.y + (cb.height - img.getHeight) / 2) -> Picture.image(img)
    draw(pic)
}

