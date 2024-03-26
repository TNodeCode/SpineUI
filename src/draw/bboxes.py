from PIL import Image, ImageDraw


class BBoxDrawer:
    """
    This class provides methods for drawing bounding boxes in images
    """
    @staticmethod
    def draw_bboxes(image: Image, bboxes: list[dict], min_score=0.5) -> Image:
        """
        Draw bounding boxes into PIL Image object

        Parameters:
            image: PIL Image object
            df: data frame containing the bounding boxes

        Return:
            Image with bounding boxes
        """
        # Create a new image with bounding boxes drawn
        draw = ImageDraw.Draw(image)

        # Define color mappings for different classes
        color_mappings = {
            'class1': '#00ff00',
            'class2': 'blue',
            'class3': 'green',
            # Add more class-color mappings as needed
        }

        # Draw bounding boxes and labels on the image
        for box in bboxes:
            if box['score'] < min_score:
                continue
            class_name = box['class_name']
            xmin = box['xmin']
            ymin = box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']
            color = color_mappings.get(class_name, 'yellow')  # Default to yellow if class color not defined
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)

        return image

