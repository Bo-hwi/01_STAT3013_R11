import numpy as np
import pandas as pd
from src.decision import matrix_report_figures as figures


def test_heatmap_labels_inside_canvas_with_manual_layout():
    families=['GROCERY I','BEVERAGES','CLEANING','DAIRY','BREAD/BAKERY','PERSONAL CARE','FROZEN FOODS']
    cities=['Long City Label '+str(i) for i in range(21)]+['Santo Domingo']
    frame=pd.MultiIndex.from_product([families,cities,['S1','S3','S5']],names=['family','city','scenario']).to_frame(index=False)
    frame['relative_delta_pct_P4_M3_minus_M0']=np.linspace(-1,1,len(frame))
    fig=figures.heatmap(frame,'P4_M3_minus_M0',families,cities,['S1','S3','S5'])
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for ax in fig.axes[:3]:
        for label in ax.get_xticklabels()+ax.get_yticklabels():
            box=label.get_window_extent(renderer)
            assert box.x0>=0 and box.y0>=0 and box.x1<=width and box.y1<=height
    figures.plt.close(fig)
