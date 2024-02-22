import umbridge


def fun():
    model = umbridge.HTTPModel("http://localhost:4242", "genz")
    print(model.get_input_sizes())
    print(model.get_output_sizes())
    sample = [[0, 0.5, 1.0, 0.7, 0.4]]
    print(model(sample))


if __name__ == '__main__':
    fun()
