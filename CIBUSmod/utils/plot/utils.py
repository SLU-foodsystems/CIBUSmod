import numpy as np
from cycler import cycler

def SetColorCycle(ax, N, cmap, reverse_cmap):
    if cmap.N==256:
        if reverse_cmap:
            cmap_range = np.linspace(0.1, 0.9, N)
        else:
            cmap_range = np.linspace(0.9, 0.1, N)
        ax.set_prop_cycle(cycler('color', cmap(cmap_range)))
    else:
        ax.set_prop_cycle(cycler('color', cmap.colors))

# Text Wrapping
# Author: user65 (https://stackoverflow.com/questions/4018860/text-box-with-line-wrapping)
# Defines wrapText which will attach an event to a given mpl.text object,
# wrapping it within the parent axes object.
# Some adjustments from original has been made.
def _wrapOneText(text, margin_px):
    """Wraps a single mpl.text object within its parent axes. margin_px is
    the margin in pixels (already converted from points)."""
    def _width(s):
        """Gets the length of a string in pixels."""
        text.set_text(s)
        return text.get_window_extent().width

    ax = text.axes

    # Find available space
    clip = ax.get_window_extent()
    x0, y0 = text.get_transform().transform(text.get_position())
    if text.get_horizontalalignment() == 'left':
        width = clip.x1 - x0 - margin_px
    elif text.get_horizontalalignment() == 'right':
        width = x0 - clip.x0 - margin_px
    else:
        width = (min(clip.x1 - x0, x0 - clip.x0) - margin_px) * 2

    # Wrap the text string
    words = [''] + _splitText(text.get_text())[::-1]
    wrapped = []

    line = words.pop()
    while words:
        line = line if line else words.pop()
        lastLine = line

        while _width(line) <= width:
            if words:
                lastLine = line
                line += words.pop()
                # Add in any whitespace since it will not affect redraw width
                while words and (words[-1].strip() == ''):
                    line += words.pop()
            else:
                lastLine = line
                break

        wrapped.append(lastLine)
        line = line[len(lastLine):]
        if not words and line:
            wrapped.append(line)

    # Set new text dropping any duplicate line breaks
    text.set_text('\n'.join(wrapped).replace('\n\n', '\n'))

def wrapText(text, margin=4):
    """ Attaches an on-draw event to a given mpl.text object which will
        automatically wrap its string wthin the parent axes object.

        The margin argument controls the gap between the text and axes frame
        in points.

        All texts registered for the same figure (e.g. one call per subplot
        title) share a single draw_event handler and trigger only one
        corrective full-figure redraw. Attaching a separate handler per text
        would each trigger its own full-figure redraw, which scales
        quadratically with the number of texts in the figure (e.g. subplot
        titles in a large subplot grid).
    """
    if text is None:
        return
    ax = text.axes
    if ax is None:
        return
    fig = ax.figure
    margin_px = margin / 72 * fig.get_dpi()

    registry = getattr(fig, '_wrapText_registry', None)
    if registry is None:
        registry = []
        fig._wrapText_registry = registry
    registry.append((text, margin_px))

    if getattr(fig, '_wrapText_connected', False):
        return
    fig._wrapText_connected = True

    def _wrap_all(event):
        """Wraps all registered texts within their parent axes."""
        for t, m in fig._wrapText_registry:
            _wrapOneText(t, m)

        # Draw wrapped strings after disabling events to prevent recursion
        handles = fig.canvas.callbacks.callbacks[event.name]
        fig.canvas.callbacks.callbacks[event.name] = {}
        fig.canvas.draw()
        fig.canvas.callbacks.callbacks[event.name] = handles

    fig.canvas.mpl_connect('draw_event', _wrap_all)

def wrapXTicks(xticks, margin=4):
    """ Adaptation of 'wrapText' for XTicks.
        Assumes center alignment and allows tick
        labels to flow out a bit on the sides.
    """
    if xticks is None or xticks[0] is None:
        return
    ax = xticks[0].axes
    margin = margin / 72 * ax.figure.get_dpi()
    
    locs = []
    texts = []
    width_fracs = []
    tot_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    for i in range(len(xticks)):
        
        locs.append(xticks[i].get_loc())

        if xticks[i].label1.get_visible():
            texts.append(xticks[i].label1)
        else:
            texts.append(xticks[i].label2)
        
        if len(xticks) > 1:
            if i>0:
                if i<(len(xticks)-1):
                    width_fracs.append(min(xticks[i].get_loc() - xticks[i-1].get_loc(), xticks[i+1].get_loc() - xticks[i].get_loc()) / tot_range)
                else:
                    width_fracs.append((xticks[i].get_loc() - xticks[i-1].get_loc()) / tot_range)
            else:
                width_fracs.append((xticks[i+1].get_loc() - xticks[i].get_loc()) / tot_range)
        else:
            width_fracs = [1.0]
    
    def _wrap(event):
        """Wraps text within its parent axes."""
        
        wrapped_texts = []
        
        for i, text in enumerate(texts):
            def _width(s):
                """Gets the length of a string in pixels."""
                text.set_text(s)
                return text.get_window_extent().width

            # Find available space
            width = ax.get_window_extent().width * width_fracs[i]
            
            # Wrap the text string
            words = [''] + _splitText(text.get_text())[::-1]
            wrapped = []
    
            line = words.pop()
            while words:
                line = line if line else words.pop()
                lastLine = line
                
                while _width(line) <= width:
                    if words:
                        lastLine = line
                        line += words.pop()
                        # Add in any whitespace since it will not affect redraw width
                        while words and (words[-1].strip() == ''):
                            line += words.pop()
                    else:
                        lastLine = line
                        break
    
                wrapped.append(lastLine)
                line = line[len(lastLine):]
                if not words and line:
                    wrapped.append(line)

            # Set new text dropping any duplicate line breaks
            text.set_text('\n'.join(wrapped).replace('\n\n','\n'))
            wrapped_texts.append('\n'.join(wrapped).replace('\n\n','\n'))

        ax.set_xticks(locs, wrapped_texts)
        
    ax.figure.canvas.mpl_connect('draw_event', _wrap)

def _splitText(text):
    """ Splits a string into its underlying chucks for wordwrapping.  This
        mostly relies on the textwrap library but has some additional logic to
        avoid splitting latex/mathtext segments.
    """
    import textwrap
    import re
    math_re = re.compile(r'(?<!\\)\$')
    textWrapper = textwrap.TextWrapper()

    if len(math_re.findall(text)) <= 1:
        return textWrapper._split(text)
    else:
        chunks = []
        for n, segment in enumerate(math_re.split(text)):
            if segment and (n % 2):
                # Mathtext
                chunks.append('${}$'.format(segment))
            else:
                chunks += textWrapper._split(segment)
        return chunks