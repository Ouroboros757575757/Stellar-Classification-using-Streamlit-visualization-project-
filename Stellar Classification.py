import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as mp
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import warnings 
warnings.filterwarnings("ignore")
mp.style.use("dark_background")
st.set_page_config(page_title="Stellar Classification", layout="wide")
st.title("Dashboard")
st.image(r"C:\Users\hp\Downloads\nebula.webp")
df=pd.read_csv(r"C:\Users\hp\Downloads\star_classification.csv (1)\star_classification.csv")
col=['obj_ID','run_ID','rerun_ID','cam_col', 'field_ID', 'spec_obj_ID', 'fiber_ID']
df.drop(col, axis=1, inplace=True)
st.info("Stellar Classification Dataframe")
df2=st.dataframe(df)

mp.title("Stellar Classification")
with st.expander("Description of the Columns"):
    st.markdown("""
**The columns:**

• **alpha**: Right Ascension – sky's longitude.  
• **delta**: Declination – sky's latitude.  
• **u**: Ultraviolet brightness (magnitude).  
• **g**: Green light brightness.  
• **r**: Red light brightness.  
• **i**: Near-infrared brightness.  
• **z**: Infrared brightness.  
• **cam_col**: Camera column used for imaging.  
• **class**: Category of the object (e.g., STAR, GALAXY, QSO).  
• **redshift**: A measure of how far the object is (higher = farther).  
• **plate**: ID of the spectroscopic plate used.  
• **MJD**: Modified Julian Date when the data was recorded.  
""")

tab1, tab2, tab3, tab4 = st.tabs(["Numeric","Distribution", "Categorical", "Map"])
num_type=st.sidebar.radio("Choose the numeric plot type: ", ["Select", "Line Plot","Heatmap", "Scatter Plot" ])
with tab1:
    if num_type=="Heatmap":
        corr_type=st.sidebar.radio("Choose the correlation map: ", ["Select","Redshift and Filters", "Spatial and Redshift"])  
        if corr_type=="Redshift and Filters":
            for c in df['class'].unique():
                sub = df[df['class'] == c]
                corr = sub[['u', 'g', 'r', 'i', 'z', 'redshift']].corr()
                fig, ax = mp.subplots()
                st.subheader(f"Correlation for Class: {c}")
                sns.heatmap(corr, annot=True, cmap='magma', ax=ax)
                st.pyplot(fig)

            with st.expander("**Analysis of the Plot**"):
                st.markdown(f""" 
    **For Galaxy Class**       
    Redshift and z:
    The moderate positive correlation of 0.77 between redshift and the z band indicates that as the redshift of galaxies increases, the flux in the z band also increases. This suggests that distant galaxies tend to have a brighter flux in the redder wavelength bands, consistent with cosmological redshift.

    Redshift and i:
    The stronger correlation of 0.81 between redshift and the i band shows that, as galaxies are observed at greater redshifts, the flux in the i band also increases significantly. This reinforces the idea that, with redshift, light from distant galaxies is shifted towards longer wavelengths, making these wavelengths more prominent in observations.

    Redshift and r:
    The highest correlation of 0.84 between redshift and the r band shows a notable relationship. This indicates that the redshifted light of galaxies is contributing more flux to the r band, which is expected because the r band is a part of the red spectrum, and galaxies at high redshifts are more influenced by the cosmological expansion.

    Redshift and g:
    The correlation of 0.83 between redshift and the g band suggests a similar behavior as r, where redshift leads to increased flux in this band. Although the g band is more sensitive to blue light, galaxies still exhibit a positive flux increase with redshift, although to a slightly lesser extent than the redder filters.

    Redshift and u:
    The lowest correlation of 0.67 with the u band shows the weakest relationship, which is expected. As redshift increases, the blue wavelengths (like u) are stretched beyond detection limits, leading to a decrease in their observable flux and resulting in a weaker correlation with redshift.
                                    
    **For Quasars**

    Quasars show weak correlations between redshift and filter magnitudes (u, g, r, i, z). The light from quasars is heavily influenced by non-thermal processes, such as synchrotron radiation and emission from the accretion disk around the supermassive black hole. These emissions are not primarily affected by the cosmological redshift in the same way as the light from galaxies.

    The weak correlations suggest that redshift does not strongly affect the observed flux in these bands, likely because quasars have other dominant emission processes that overshadow the typical redshift effect.

    The highest correlation with the u band (0.32) suggests that quasars' light in the blue region is somewhat more responsive to redshift, though still weaker than the correlations seen in galaxies.
                                    
    **For Stars**
                                    
    For stars, the negative and very weak correlations between redshift and the filter magnitudes suggest that redshift does not influence the flux in any observable way across the optical and ultraviolet bands. This is likely because:

    Most stars are relatively close to Earth, meaning their redshift values are small. For nearby stars, the effects of cosmological redshift are not noticeable in their observed flux.

    Stars' light is not strongly impacted by redshift unless they are at extremely high redshifts (which is not typically the case for most stars). Therefore, we do not see a clear relationship between redshift and the magnitudes in these filters.

    The negative correlations being close to zero further reinforce that there is almost no effect of redshift on the observed magnitudes of stars, which is expected unless we are dealing with very distant stars (e.g., stars in distant galaxies or quasars) that experience significant cosmological redshift.
                                  """)
    
        elif corr_type=="Spatial and Redshift":
            cols=['alpha', 'delta', 'redshift']
            corr=df[cols].corr()
            fig, ax= mp.subplots()
            sns.heatmap(corr, annot=True, cmap="magma")
            st.pyplot(fig)
            with st.expander(f"**Analysis of the Plot**"):
                st.markdown(f""" The Spatial and Redshift heatmap shows the correlation between the spatial coordinates (RA and DEC, represented as alpha and delta) and the redshift (redshift). The spatial coordinates (alpha and delta) represent the position of celestial objects in the sky (right ascension and declination), while the redshift (redshift) indicates the relative velocity and distance of those objects.
                                    The correlation between redshift and alpha (RA) tells us how the right ascension of the celestial object is related to its redshift. 
                                    Generally, we don't expect a very strong relationship between the spatial position (RA) and redshift, as the redshift is more related to the object's velocity and distance rather than its position in the sky. The correlation is close to zero or negative, this suggests that there isn't a strong spatial dependency between the right ascension (alpha) and redshift.
                                    """)
                    
    elif num_type=="Line Plot":
        df['jd']=df['MJD'].values + 2400000.5
        jd = df['jd'].values
        t = Time(jd, format='jd')
        gd= t.iso
        datedf=pd.Series(gd)
        date1=pd.to_datetime(datedf)
        df['date']=date1.dt.date
        df['date']=pd.to_datetime(df['date'])
        df['year']=df['date'].dt.year
        redshift_grp=df.groupby('year')['redshift'].median().reset_index()
        fig, ax=mp.subplots()
        sns.lineplot(data=redshift_grp, x="year", y="redshift", marker='*')
        years=np.arange(2000, 2021, 1)
        mp.xticks(years, rotation=90)
        mp.grid(True, linestyle='--')
        st.pyplot(fig)
        with st.expander(f"**Analysis of the Plot**"):
            st.markdown(f""" The median redshift per year plot gives us valuable insights into how the distance of observed objects has changed over time. 
                                The use of the median is particularly useful in handling any outliers or extreme redshift values, ensuring that the plot reflects the central tendency of the data accurately. 
                                Depending on the trends observed, this could provide a better understanding of how the characteristics of objects in the dataset have evolved over time.

    **Initial Decrease in Redshift**: The redshift decreases from 0.1 to 0.0 in the early years. This may indicate that in the beginning, the observations were dominated by relatively closer objects with lower redshift values. 
    This could also reflect changes in observational strategies, the focus on nearby galaxies, or the dataset's evolution.

    **Constant Median Redshift**: From the 0.0 value, the median redshift remains constant for several years, hovering around 0.5 for about four years.
    This could imply that the objects observed during this period were more consistent in their redshift values, potentially representing a specific class of objects (e.g., objects within a similar distance range, or observational focus).

    **Fluctuations**: The plot then shows an increase in redshift to 0.9, followed by fluctuations between 0.7 and 0.9, and then a slight decrease to 0.6. These fluctuations suggest that the redshift values of the observed objects became more variable in later years. 
    It could indicate a shift in the type of objects being observed (e.g., a mix of nearer and more distant objects, or different kinds of celestial bodies like galaxies and quasars with varying redshifts). These variations may also reflect advancements in observation technology, more extensive surveys, or an increasing number of distant objects being discovered.
                                """)
                    
    elif num_type=="Scatter Plot":
        df['jd']=df['MJD'].values + 2400000.5
        jd = df['jd'].values
        t = Time(jd, format='jd')
        gd= t.iso
        datedf=pd.Series(gd)
        date1=pd.to_datetime(datedf)
        df['date']=date1.dt.date
        df['date']=pd.to_datetime(df['date'])
        df['year']=df['date'].dt.year
        fig, ax= mp.subplots()
        sns.scatterplot(data=df, x="year", y="redshift", marker='*')
        years=np.arange(2000,2021,1)
        mp.xticks(years, rotation=90)
        mp.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        with st.expander(f"**Analysis of the Plot**"):
                    st.markdown(f""" 
    **Steady Redshift Trend Over Time**: The scatter plot shows redshift values plotted against the years. Each marker represents a redshift value for a specific year.
    The markers are arranged in a straight line, suggesting that for each year, the redshift values consistently fall within a specific range (between 1 and 7).
    Given the range of redshift values between 1 and 7, it suggests that the objects in the dataset are at varying distances, likely corresponding to different redshifts (higher values representing greater distances). 
    Despite this, the markers align in a straight line, which could indicate a relatively uniform redshift pattern across the years in the data.                           
                                """)                
with tab2:        
    dis_type=st.sidebar.radio("Select the distribution plot: ", ["Select","Histogram", "KDE", "Violin Plot"])
    if dis_type=="Histogram":
        st.title("Redshift Distribution")
        clr=st.color_picker("pick a color")
        fig, ax=mp.subplots()
        sns.histplot(data=df, x="redshift",ax=ax, bins=17, color=clr, kde=True)
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Count of the objects")
        st.pyplot(fig)
        with st.expander(f"**Analysis of {dis_type} Plot**"):
            st.markdown(""" 
**Majority of Objects are Closer**:
The tall bars on the left side of the histogram suggest that most of the objects in the dataset have relatively low redshift values. 
Since lower redshift values correspond to objects that are closer to Earth, this indicates that the majority of the objects in the dataset are relatively near to us in cosmic terms.

**Left-Skewed Distribution**:
The histogram seems to be left-skewed, meaning that the bulk of the objects are clustered near lower redshift values, with fewer objects at higher redshifts. 
This is typical in many astronomical surveys where closer objects (such as nearby galaxies or stars) are more numerous and easier to detect.

**Possible Explanation**:
This could be due to the nature of astronomical surveys, where nearer objects are often more abundant, and distant objects (higher redshifts) are more difficult to observe. 
Furthermore, cosmic expansion causes redshift to increase with distance, so the number of objects at higher redshifts generally decreases as you look further away in the universe.
                            """)
                
    elif dis_type=="KDE":
        ste_type=st.sidebar.radio("Select the class: ", ["Select", "Galaxy", "Star", "Quasar"])
        if ste_type=="Galaxy":
            st.title("Redshift Distribution among Galaxy")
            fig, ax= mp.subplots()
            sns.kdeplot(data=df[df["class"] == "GALAXY"], x="redshift", fill=True)
            st.pyplot(fig)
            with st.expander(f"**Analysis of the Plot**"):
                st.markdown(""" 
There are multiple peaks, suggesting a multimodal distribution (i.e., your redshift values are clustered in different ranges).

The first peak is around 0.1–0.2, meaning that range has the highest concentration of redshift values.

Another small hump appears around 0.4–0.5, indicating a second concentration.

After 1.0, the density tapers off — meaning high redshift values are rare.""")
                
        elif ste_type=="Star":
            st.title("Redshift Distribution among Star")
            fig, ax= mp.subplots()
            sns.kdeplot(data=df[df["class"] == "STAR"], x="redshift", fill=True)
            st.pyplot(fig)
            with st.expander(f"**Analysis of the Plot**"):
                st.markdown(f""" 
The x-axis values are tightly centered around 0 (looks like something close to a mean or difference value).

The y-axis (density) reaches over 1400, which is very high — indicating the data is very concentrated.

The curve is super narrow, with values mostly between -0.004 and +0.004.
                            """)
                
        elif ste_type=="Quasar":
            st.title("Redshift Distribution among Quasar")
            fig, ax= mp.subplots()
            sns.kdeplot(data=df[df["class"] == "QSO"], x="redshift", fill=True)
            st.pyplot(fig)
            with st.expander(f"**Analysis of the Plot**"):
                st.markdown(f""" 
Data is bounded at 0 (or close), and then stretches out positively.

The distribution is not symmetrical.

If this is a feature from a dataset, it probably needs scaling or transformation (like log-scaling) before being used in many ML models.
                            """)
                
    elif dis_type=='Violin Plot':
        fig, ax= mp.subplots()
        st.title("Classes and redshift")
        sns.violinplot(x=df["class"], y=df['redshift'], palette="pastel")
        st.pyplot(fig)
        with st.expander(f"**Analysis of {dis_type} Plot**"):
            st.markdown(f""" 
This plot shows how redshift values are distributed for each object class — galaxy, quasar, and star — and helps compare their spread, median, and density.
The violin shape for stars is very narrow and centered around redshift ≈ 0, indicating that stars typically have very low redshift values. This makes sense, as stars are usually much closer compared to galaxies and quasars.
The distribution for galaxies is more spread out, usually starting near 0 and extending up to moderate redshift values. The wider parts of the violin show where redshift values are more frequent. This reflects the presence of both nearby and moderately distant galaxies in the dataset.
Quasars show a wider and more skewed distribution toward higher redshift. This suggests that quasars in the dataset are typically much farther than galaxies and stars — which aligns with their nature as very distant and luminous objects.
                                                            """)
                
with tab3:
    cat_type=st.sidebar.radio("Select the categorical plot: ", ["Select", "Pie Plot", "Bar Plot"])   
    if cat_type=="Pie Plot":
        st.title("Class Distribution")
        clr1=st.color_picker("pick a color", key="color1")
        clr2=st.color_picker("pick a color", key="color2")
        clr3=st.color_picker("pick a color", key="color3")
        mp.figure(figsize=(10,5))
        fig, ax= mp.subplots()
        mp.pie(df["class"].value_counts(), autopct='%1.1f%%',labels=df['class'].value_counts().index, colors=[clr1, clr2, clr3])
        st.pyplot(fig)
        with st.expander(f"**Analysis of {cat_type} Plot**"):
            st.markdown(f""" 
The pie chart reveals how the dataset is divided among different celestial object classes. This gives an overall view of class distribution, helping identify potential class imbalance, which is important for further analysis or machine learning tasks.
                                """)
        
    elif cat_type=='Bar Plot':
        st.title("Stellar Objects Classification")
        clr=st.color_picker("pick a color")
        mp.figure(figsize=(10,5))
        fig, ax = mp.subplots()
        sns.countplot(df['class'], color=clr)
        st.pyplot(fig)
        with st.expander("**Analysis of Bar Plot**"):
            st.markdown(""" 
The bar plot effectively highlights the distribution of celestial object classes, showing which type is most commonly recorded.
This aids in understanding the focus of the dataset and planning further class-specific analysis.
                                """)
with tab4:
    galaxy=df[df['class']=='GALAXY']
    quasars=df[df['class']=='QSO']
    stars=df[df['class']== 'STAR']
    gal1=galaxy['delta'].values*u.deg
    gal2=galaxy['alpha'].values*u.deg
    gal_sky= SkyCoord(ra= gal2, dec=gal1, frame='icrs')
    fig, ax= mp.subplots(subplot_kw={"projection":"aitoff"})
    ax.scatter(np.radians(gal_sky.ra.deg), np.radians(gal_sky.dec.deg), marker='*')
    ax.grid(True)
    mp.title("2-D sky map for galaxies")
    mp.tight_layout()
    st.pyplot(fig)
    star1=stars['alpha'].values*u.deg
    star2=stars['delta'].values*u.deg
    star_plot=SkyCoord(ra=star1, dec=star2, frame='icrs')
    fig, ax=mp.subplots(subplot_kw={'projection':'aitoff'})
    mp.scatter(np.radians(star_plot.ra.deg), np.radians(star_plot.dec.deg), marker='*')
    mp.grid(True)
    mp.title('Sky map for stars observations')
    st.pyplot(fig)
    qso1=quasars['alpha'].values*u.deg
    qso2=quasars['delta'].values*u.deg
    qso_plot=SkyCoord(ra=qso1, dec=qso2, frame='icrs')
    fig, ax=mp.subplots(subplot_kw={'projection':'aitoff'})
    mp.scatter(np.radians(qso_plot.ra.deg), np.radians(qso_plot.dec.deg), marker='*')
    mp.grid(True)
    mp.title('Sky map for quasars observations')
    st.pyplot(fig)


