import torchvision


class Transformer:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def getAllImages(self, transforms, scale_sizes):
        trans_images, trans_labels = self.transformedImgaes(transforms)
        scale_images, scale_labels = self.scaledImages(scale_sizes)
        trans_images.extend(scale_images)
        trans_labels.extend(scale_labels)
        return trans_images, trans_labels

    def transformedImgaes(self, transforms):
        result_images = []
        result_labels = []
        for image, label in zip(self.images, self.labels):
            image = torchvision.transforms.ToPILImage()(image)
            image = torchvision.transforms.Resize((224, 224))(image)
            result_images.append(image)
            result_labels.append(label)
            for transform in transforms:
                result_images.append(transform(image))
                result_labels.append(label)
        return result_images, result_labels

    def scaledImages(self, scale_sizes):
        result = []
        result_labels = []
        for image, label in zip(self.images, self.labels):
            for size in scale_sizes:
                pil_image = torchvision.transforms.ToPILImage()(image)
                pil_image = torchvision.transforms.Resize(size)(pil_image)
                pil_image = torchvision.transforms.CenterCrop((224, 224))(pil_image)
                result.append(pil_image)
                result_labels.append(label)
        return result, result_labels

