// #include /nn.kojo

// y = 3 x^2
// dy/dx = 6 x
// at x = 4, y = 48, dy/dx = 24

def y(x: NDArray): NDArray = {
    x.pow(2).mul(3)
}

ndScoped { use =>
    val nm = use(ndMaker)
    val gc = use(gradientCollector)
    val x1 = nm.create(4f)
    x1.setRequiresGradient(true)
    val y1 = y(x1)
    gc.backward(y1)
    println(x1)
    println(y1)
    println(x1.getGradient)
}
