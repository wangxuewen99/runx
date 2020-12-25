import torch

def single_draw_box(image, boxes, color, valid_flags=None, format='CHW'):
    if valid_flags is not None:
        if valid_flags.any():
            boxes = boxes[valid_flags]
        else:
            return image

    if format == 'CHW':
        height, width = image.shape[-2:]
    elif format == 'HWC':
        height, width = image.shape[:2]
    else:
        raise Exception('invalid data format.')

    boxes = boxes.long()
    boxes[:, 0] = torch.clamp(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = torch.clamp(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = torch.clamp(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = torch.clamp(boxes[:, 3], 0, height - 1)
    color = image.new_tensor(color).view(3, 1)

    for box in boxes:
        x0, y0, x1, y1 = box
        if format == 'CHW':
            image[:, y0, x0: x1] = color    # draw top line
            image[:, y1, x0: x1] = color    # draw bottom line
            image[:, y0: y1, x0] = color    # draw left line
            image[:, y0: y1, x1] = color    # draw right line
        elif format == 'HWC':
            image[y0, x0: x1, :] = color    # draw top line
            image[y1, x0: x1, :] = color    # draw bottom line
            image[y0: y1, x0, :] = color    # draw left line
            image[y0: y1, x1, :] = color    # draw right line
        else:
            raise Exception('invalid data format.')
    return image

def draw_box(image, boxes, color, valid_flags=None, format='CHW'):
    if image.dim() == 3 and len(format) == 3:
        single_draw_box(image, boxes, color, valid_flags, format)
    elif image.dim() == 4 and len(format) == 4:
        batch_size = image.size(0)
        for k in range(batch_size):
            valid_flag = None if valid_flags is None else valid_flags[k]
            single_draw_box(image[k], boxes[k], color, valid_flag, format[1:])
    else:
        raise Exception('invalid image data format.')

    return image

def single_draw_point(image, pts, color, valid_flags=None, format='CHW'):
    if valid_flags is not None:
        if valid_flags.any():
            pts = pts[valid_flags]
        else:
            return image

    shape = image.shape
    if format == 'CHW':
        channel, height, width = shape
    elif format == 'HWC':
        height, width, channel = shape
    else:
        raise Exception('invalid image data format.')

    pts = 0.5*(pts + 1)*torch.tensor([[width, height]], device=pts.device)
    pts = pts.long()
    x = torch.clamp(pts[:, 0], 1, width-2)
    y = torch.clamp(pts[:, 1], 1, height-2)
    if format == 'CHW':
        indices0 = torch.cat([(y-1)*width+x, y*width+x, (y+1)*width+x, y*width+(x-1), y*width+(x+1)], dim=0)
        indices1 = indices0 + width*height
        indices2 = indices1 + width*height
    elif format == 'HWC':
        indices0 = torch.cat([(y-1)*width+x, y*width+x, (y+1)*width+x, y*width+(x-1), y*width+(x+1)], dim=0)
        indices0 = indices0 * channel
        indices1 = indices0 + 1
        indices2 = indices0 + 2

    image = image.reshape(-1)
    r, g, b = color
    image.index_fill_(dim=0, index=indices0, value=r)
    image.index_fill_(dim=0, index=indices1, value=g)
    image.index_fill_(dim=0, index=indices2, value=b)
    image = image.reshape(shape)
    return image

def draw_point(image, pts, color, valid_flags=None, format='CHW'):
    if image.dim() == 3 and len(format) == 3:
        single_draw_point(image, pts, color, valid_flags, format)
    elif image.dim() == 4 and len(format) == 4:
        batch_size = image.size(0)
        for k in range(batch_size):
            valid_flag = None if valid_flags is None else valid_flags[k]
            single_draw_point(image[k], pts[k], color, valid_flag, format[1:])
    else:
        raise Exception('invalid image data format.')

    return image


