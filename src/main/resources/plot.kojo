import net.kogics.kojo.plot._

import org.knowm.xchart.internal.chartpart.Chart
import org.knowm.xchart.internal.series.Series
import org.knowm.xchart.style.Styler

var currChartPic: Option[Picture] = None

def chartPic[A <: Styler, B <: Series](chart: Chart[A, B]): Picture = {
    import org.knowm.xchart.BitmapEncoder
    val img = BitmapEncoder.getBufferedImage(chart)
    Picture.image(img)
}

def centeredChartPic[A <: Styler, B <: Series](chart: Chart[A, B]): Picture = {
    val cb = canvasBounds
    val pic = chartPic(chart)
    val pb = pic.bounds
    pic.withTranslation(
        cb.x + (cb.width - pb.width) / 2,
        cb.y + (cb.height - pb.height) / 2
    )
}

def drawChart[A <: Styler, B <: Series](chart: Chart[A, B]) {
    currChartPic.foreach { pic =>
        pic.erase()
    }
    val newPic = centeredChartPic(chart)
    currChartPic = Some(newPic)
    draw(newPic)
}

def updateChart[A <: Styler, B <: Series](chart: Chart[A, B]) {
    drawChart(chart)
}
