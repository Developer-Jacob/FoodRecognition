import torchvision


def scale(image, size):
    image = torchvision.transforms.Resize(size)(image)
    image = torchvision.transforms.CenterCrop((224, 224))(image)
    return image


def transformer(images, labels, transforms, scale_sizes=[256]):
    result_images = []
    result_labels = []
    for image, label in zip(images, labels):
        result_image = torchvision.transforms.ToPILImage()(image)
        for scale_size in scale_sizes:
            scaled_image = scale(result_image, scale_size)
            result_images.append(scaled_image)
            result_labels.append(label)
            for transform in transforms:
                result_images.append(transform(scaled_image))
                result_labels.append(label)
            del scaled_image
        del result_image
    return result_images, result_labels
