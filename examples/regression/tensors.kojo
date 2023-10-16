// #include /nn.kojo

// y = 3 x^2
// dy/dx = 6 x
// at x = 4, y = 48, dy/dx = 24

clearOutput()

def y(x: NDArray): NDArray = {
    x.pow(2).mul(3)
}

def numericalYGradient(atx: NDArray): NDArray = {
    val h = 0.001f
    val y0 = y(atx)
    val y1 = y(atx.add(h))
    y1.sub(y0).div(h)
}

ndScoped { use =>
    val nm = use(ndMaker)
    val gc = use(gradientCollector)
    val x1 = nm.create(4f)
    x1.setRequiresGradient(true)
    val y1 = y(x1)
    gc.backward(y1)
    println(s"At x=${x1.getFloat()}, y=${y1.getFloat()}, dy/dx = ${x1.getGradient.getFloat()}")
    println("---")
    val ng = numericalYGradient(x1)
    println(s"At x=${x1.getFloat()}, numerical dy/dx = ${ng.getFloat()}")
}
