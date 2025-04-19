from .header import create as create_header
from .input_panel import create as create_input_panel
from .output_panel import create as create_output_panel
from .footer import create as create_footer


def assemble_layout(state_manager):
    """组装完整界面布局"""
    header = create_header()
    input_panel = create_input_panel()
    output_panel = create_output_panel()
    footer = create_footer()

    return {
        "header": header,
        "input_panel": input_panel,
        "output_panel": output_panel,
        "footer": footer
    }