/******************************************************************************
 * sofalizer.c : SOFAlizer plugin to use SOFA files in vlc
 *****************************************************************************
 * Copyright (C) 2013 Andreas Fuchs, ARI
 *
 * Authors: Andreas Fuchs <andi.fuchs.mail@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
*****************************************************************************/

/*****************************************************************************
 * Preamble
 *****************************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <vlc_common.h>
#include <vlc_plugin.h>
#include <vlc_aout.h>
#include <vlc_filter.h>
#include <vlc_block.h>
#include <vlc_modules.h>

#include <math.h>
#include <netcdf.h>

#define N_SOFA 3
#define N_POSITIONS 4

/*****************************************************************************
 * Local prototypes
 *****************************************************************************/

struct nc_sofa_t
{
   int i_ncid;
   int i_n_samples;
   int i_m_dim;
   int *p_data_delay;
   float *p_sp_a;
   float *p_sp_e;
   float *p_sp_r;
   float *p_data_ir;
};

struct filter_sys_t
{
    struct nc_sofa_t sofa[N_SOFA];
    vlc_mutex_t lock;

    float *p_speaker_pos;

    /* N of Channels to convolute */
    int i_n_conv;

    /* Buffer variables */
    float *p_ringbuffer_l;
    float *p_ringbuffer_r;
    int i_write;
    int i_buffer_length;

    /* NetCDF variables */
    int i_i_sofa;  /* Selected Sofa File */
    int *p_delay_l;
    int *p_delay_r;
    float *p_ir_l;
    float *p_ir_r;

    /* Control variables */
    float f_gain;
    float f_rotation;
    float f_elevation;
    float f_radius;
    int i_azimuth_array[4];
    int i_elevation_array[4];
    int i_switch;
    bool b_mute;

    bool b_lfe;

};

struct t_thread_data
{
    filter_sys_t *p_sys;
    block_t *p_in_buf;
    int *p_input_nb;
    int *p_delay;
    int i_write;
    int *p_n_clippings;
    float *p_ringbuffer;
    float *p_dest;
    float *p_ir;
    float f_gain_lfe;
};

struct data_findM_t
{
    filter_sys_t *p_sys;
    int i_azim;
    int i_elev;
    int *p_m;
    float f_radius;
};

static int  Open ( vlc_object_t *p_this );
static void Close( vlc_object_t * );
static block_t *DoWork( filter_t *, block_t * );

static int LoadIR ( filter_t *p_filter, int i_azim, int i_elev, float f_radius);
void sofalizer_Convolute ( void *data );
void sofalizer_FindM ( void *data );

#define DECLARECB(fn) static int fn (vlc_object_t *,char const *, \
                                     vlc_value_t, vlc_value_t, void *)
DECLARECB( GainCallback  );
DECLARECB( RotationCallback   );
DECLARECB( ElevationCallback   );
DECLARECB( SelectCallback  );
DECLARECB( RadiusCallback );
DECLARECB( SwitchCallback );

#undef  DECLARECB

/*****************************************************************************
 * Module descriptor
 *****************************************************************************/

#define HELP_TEXT N_( "SOFAlizer creates a virtual auditory display, i.e., virtual loudspeakers around the user for listening via headphones. The position of the virtual loudspeakers depends on the audio format of the input file (up to 8.1 supported). SOFAlizer filters audio channels with head-related transfer functions (HRTFs) stored in SOFA files (www.sofaconventions.org) following the SimpleFreeFieldHRIR Convention. A database of SOFA files can be found at www.sofacoustics.org.\nSOFAlizer is developed at the Acoustics Research Institute (ARI) of the Austrian Academy of Sciences." )

#define GAIN_VALUE_TEXT N_( "Gain [dB]" )
#define GAIN_VALUE_LONGTEXT N_( "Sets the gain of the module." )

#define FILE1_NAME_TEXT N_( "SOFA file 1" )
#define FILE2_NAME_TEXT N_( "SOFA file 2" )
#define FILE3_NAME_TEXT N_( "SOFA file 3" )

#define FILE_NAME_LONGTEXT N_( "The sampling rate of the different files must equal to the sampling rate of the first (loaded) file." )

#define SELECT_VALUE_TEXT N_( "Select SOFA file" )
#define SELECT_VALUE_LONGTEXT N_( "SOFAlizer allows to load 3 different SOFA files and easily switch between them for better comparison." )

#define ROTATION_VALUE_TEXT N_( "Rotation [°]" )
#define ROTATION_VALUE_LONGTEXT N_( "Rotates virtual loudspeakers." )

#define ELEVATION_VALUE_TEXT N_( "Elevation [°]" )
#define ELEVATION_VALUE_LONGTEXT N_( "Elevates the virtual loudspeakers." )

#define RADIUS_VALUE_TEXT N_( "Radius [m]")
#define RADIUS_VALUE_LONGTEXT N_( "Varies the distance between the loudspeakers and the listener with near-field HRTFs." )

#define SWITCH_VALUE_TEXT N_( "Switch" )
#define SWITCH_VALUE_LONGTEXT N_( "Presents all audio channels from one of four pre-defined virtual positions. Position 0 activates Rotation and Elevation." )

#define POS_VALUE_LONGTEXT N_( "Only active for Switch 1-4." )

#define POS1_AZIMUTH_VALUE_TEXT N_( "Azimuth Position 1 ")
#define POS1_ELEVATION_VALUE_TEXT N_( "Elevation Position 1 ")
#define POS2_AZIMUTH_VALUE_TEXT N_( "Azimuth Position 2 ")
#define POS2_ELEVATION_VALUE_TEXT N_( "Elevation Position 2 ")
#define POS3_AZIMUTH_VALUE_TEXT N_( "Azimuth Position 3 ")
#define POS3_ELEVATION_VALUE_TEXT N_( "Elevation Position 3 ")
#define POS4_AZIMUTH_VALUE_TEXT N_( "Azimuth Position 4 ")
#define POS4_ELEVATION_VALUE_TEXT N_( "Elevation Position 4 ")

vlc_module_begin ()
    set_description( N_("SOFAlizer") )
    set_shortname( N_("SOFAlizer") )
    set_capability( "audio filter", 0)
    set_help( HELP_TEXT )
    add_loadfile( "sofalizer-filename1", "", FILE1_NAME_TEXT, FILE_NAME_LONGTEXT, false)
    add_loadfile( "sofalizer-filename2", "", FILE2_NAME_TEXT, FILE_NAME_LONGTEXT, false)
    add_loadfile( "sofalizer-filename3", "", FILE3_NAME_TEXT, FILE_NAME_LONGTEXT, false)
    add_float_with_range( "sofalizer-select", 1 , 1 , 3,  SELECT_VALUE_TEXT, SELECT_VALUE_LONGTEXT, false)
    add_float_with_range( "sofalizer-gain", 0.0, -20, 40,  GAIN_VALUE_TEXT, GAIN_VALUE_LONGTEXT, false )
    add_float_with_range( "sofalizer-rotation", 0, -360, 360, ROTATION_VALUE_TEXT, ROTATION_VALUE_LONGTEXT, false )
    add_float_with_range( "sofalizer-elevation", 0, -90, 90, ELEVATION_VALUE_TEXT, ELEVATION_VALUE_LONGTEXT, false )
    add_float_with_range( "sofalizer-radius", 1, 0, 2.1,  RADIUS_VALUE_TEXT, RADIUS_VALUE_LONGTEXT, false )
    add_float_with_range( "sofalizer-switch", 0, 0, 4, SWITCH_VALUE_TEXT, SWITCH_VALUE_LONGTEXT, false )
    add_integer_with_range( "sofalizer-pos1-azi", 90, -180, 180,POS1_AZIMUTH_VALUE_TEXT, POS_VALUE_LONGTEXT, false )
    add_integer_with_range( "sofalizer-pos1-ele", 0, -90, 90, POS1_ELEVATION_VALUE_TEXT, POS_VALUE_LONGTEXT, false )
    add_integer_with_range( "sofalizer-pos2-azi", 180, -180, 180, POS2_AZIMUTH_VALUE_TEXT, POS_VALUE_LONGTEXT, false )
    add_integer_with_range( "sofalizer-pos2-ele", 0, -90, 90, POS2_ELEVATION_VALUE_TEXT, POS_VALUE_LONGTEXT, false )
    add_integer_with_range( "sofalizer-pos3-azi", -90, -180, 180, POS3_AZIMUTH_VALUE_TEXT, POS_VALUE_LONGTEXT, false )
    add_integer_with_range( "sofalizer-pos3-ele", 0, -90, 90, POS3_ELEVATION_VALUE_TEXT, POS_VALUE_LONGTEXT, false )
    add_integer_with_range( "sofalizer-pos4-azi", 0, -180, 180, POS4_AZIMUTH_VALUE_TEXT, POS_VALUE_LONGTEXT, false )
    add_integer_with_range( "sofalizer-pos4-ele", 90, -90, 90, POS4_ELEVATION_VALUE_TEXT, POS_VALUE_LONGTEXT, false )
    add_shortcut( "sofalizer" )
    set_category( CAT_AUDIO )
    set_subcategory( SUBCAT_AUDIO_AFILTER )
    set_callbacks( Open, Close )
vlc_module_end ()

/*****************************************************************************
* CloseSofa: Closes the given Sofa file and frees it's allocated memory.
* LoadSofa: Load the Sofa files, check for the most important SOFAconventions
*     and load the whole IR Data, Source-Positions and Delays
* GetSpeakerPos: Get the Speaker Positions for current input.
* MaxDelay: Find the Maximum Delay in the Sofa File
* CompensateVolume: Compensate the Volume of the Sofa file. The Energy of the
*     IR closest to ( 0°, 0°, 1m ) to the left ear is calculated.
* FreeAllSofa: Frees Memory allocated in LoadSofa of all Sofa files
* FreeFilter: Frees Memory allocated in Open
******************************************************************************/

static int CloseSofa ( struct nc_sofa_t *sofa )
{
    free( sofa->p_data_delay );
    free( sofa->p_sp_a );
    free( sofa->p_sp_e );
    free( sofa->p_sp_r );
    free( sofa->p_data_ir );
    nc_close( sofa->i_ncid );
    sofa->i_ncid = 0;
    return VLC_SUCCESS;
}


static int LoadSofa ( filter_t *p_filter, char *c_filename, int i_i_sofa , int *p_samplingrate)
{
    struct filter_sys_t *p_sys = p_filter->p_sys;
    int i_ncid, i_n_dims, i_n_vars, i_n_gatts, i_n_unlim_dim_id, i_status;
    unsigned int i_samplingrate;
    int i_n_samples = 0;
    int i_m_dim = 0;
    p_sys->sofa[i_i_sofa].i_ncid = 0;
    i_status = nc_open( c_filename , NC_NOWRITE, &i_ncid);
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Can't find SOFA-file '%s'", c_filename);
        return VLC_EGENERIC;
    }
    nc_inq(i_ncid, &i_n_dims, &i_n_vars, &i_n_gatts, &i_n_unlim_dim_id); /* Get Number of Dimensions, Vars, Global Attributes and Id of unlimited Dimensions */

    char c_dim_names[i_n_dims][NC_MAX_NAME];   /* Get Dimensions */
    uint32_t i_dim_length[i_n_dims];
    int i_m_dim_id = 0;
    int i_n_dim_id = 0;
    for( int ii = 0; ii<i_n_dims; ii++ )
    {
        nc_inq_dim( i_ncid, ii, c_dim_names[ii], &i_dim_length[ii] );
        if ( !strcmp("M", c_dim_names[ii] ) )
            i_m_dim_id = ii;
        if ( !strcmp("N", c_dim_names[ii] ) )
            i_n_dim_id = ii;
        else { }
    }
    i_n_samples = i_dim_length[i_n_dim_id];
    i_m_dim =  i_dim_length[i_m_dim_id];

    uint32_t i_att_len;
    i_status = nc_inq_attlen(i_ncid, NC_GLOBAL, "Conventions", &i_att_len);
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Can't get Length of Attribute Conventions.");
        nc_close(i_ncid);
        return VLC_EGENERIC;
    }

    char psz_conventions[i_att_len + 1];
    nc_get_att_text( i_ncid , NC_GLOBAL, "Conventions", psz_conventions);
    *( psz_conventions + i_att_len ) = 0;
    if ( strcmp( "SOFA" , psz_conventions ) )
    {
        msg_Err(p_filter, "Not a SOFA file!");
        nc_close(i_ncid);
        return VLC_EGENERIC;
    }
    nc_inq_attlen (i_ncid, NC_GLOBAL, "SOFAConventions", &i_att_len );
    char psz_sofa_conventions[i_att_len + 1];
    nc_get_att_text(i_ncid, NC_GLOBAL, "SOFAConventions", psz_sofa_conventions);
    *( psz_sofa_conventions + i_att_len ) = 0;
    if ( strcmp( "SimpleFreeFieldHRIR" , psz_sofa_conventions ) )
    {
       msg_Err(p_filter, "No SimpleFreeFieldHRIR file!");
       nc_close(i_ncid);
       return VLC_EGENERIC;
    }

    int i_samplingrate_id;
    i_status = nc_inq_varid( i_ncid, "Data.SamplingRate", &i_samplingrate_id);
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read variable Data.SamplingRate ID");
        nc_close(i_ncid);
        return VLC_EGENERIC;
    }

    i_status = nc_get_var_uint( i_ncid, i_samplingrate_id, &i_samplingrate );
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read value of Data.SamplingRate.");
        nc_close(i_ncid);
        return VLC_EGENERIC;
    }
    *p_samplingrate = i_samplingrate;

    int i_data_ir_id;
    i_status = nc_inq_varid( i_ncid, "Data.IR", &i_data_ir_id);
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read Id of Data.IR." );
        return VLC_EGENERIC;
    }

    int *p_data_delay = p_sys->sofa[i_i_sofa].p_data_delay = calloc ( sizeof( int ) , i_m_dim * 2 );
    float *p_sp_a = p_sys->sofa[i_i_sofa].p_sp_a = malloc( sizeof(float) * i_m_dim);
    float *p_sp_e = p_sys->sofa[i_i_sofa].p_sp_e = malloc( sizeof(float) * i_m_dim);
    float *p_sp_r = p_sys->sofa[i_i_sofa].p_sp_r = malloc( sizeof(float) * i_m_dim);
    float *p_data_ir = p_sys->sofa[i_i_sofa].p_data_ir = malloc( sizeof( float ) * 2 * i_m_dim * i_n_samples );

    if ( !p_data_delay || !p_sp_a || !p_sp_e || !p_sp_r || !p_data_ir )
    {
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_ENOMEM;
    }

    i_status = nc_get_var_float( i_ncid, i_data_ir_id, p_data_ir );
    if ( i_status != NC_NOERR )
    {
        msg_Err( p_filter, "Couldn't read Data.IR!" );
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }

    int i_sp_id;
    i_status = nc_inq_varid(i_ncid, "SourcePosition", &i_sp_id);
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read ID of SourcePosition");
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }

    i_status = nc_get_vara_float (i_ncid, i_sp_id, (uint32_t[2]){ 0 , 0 } , (uint32_t[2]){ i_m_dim , 1 } , p_sp_a );
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read SourcePosition.");
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }

    i_status = nc_get_vara_float (i_ncid, i_sp_id, (uint32_t[2]){ 0 , 1 } , (uint32_t[2]){ i_m_dim , 1 } , p_sp_e );
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read SourcePosition.");
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }

    i_status = nc_get_vara_float (i_ncid, i_sp_id, (uint32_t[2]){ 0 , 2 } , (uint32_t[2]){ i_m_dim , 1 } , p_sp_r );
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read SourcePosition.");
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }

    int i_data_delay_id;
    int i_data_delay_dim_id[2];
    char i_data_delay_dim_name[NC_MAX_NAME];

    i_status = nc_inq_varid(i_ncid, "Data.Delay", &i_data_delay_id);
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read Id of Data.Delay." );
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }
    i_status = nc_inq_vardimid ( i_ncid, i_data_delay_id, &i_data_delay_dim_id[0]);
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read Dimension Ids of Data.Delay." );
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }
    i_status = nc_inq_dimname ( i_ncid, i_data_delay_dim_id[0], i_data_delay_dim_name );
    if (i_status != NC_NOERR)
    {
        msg_Err(p_filter, "Couldn't read Dimension Name of Data.Delay." );
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }
    if ( !strncmp ( i_data_delay_dim_name, "I", 1 ) )
    {
        msg_Dbg ( p_filter, "DataDelay in Dimension IR");
        int i_Delay[2];
        i_status = nc_get_var_int( i_ncid, i_data_delay_id, &i_Delay[0] );
        if ( i_status != NC_NOERR )
        {
            msg_Err(p_filter, "Couldn't read Data.Delay");
            CloseSofa( &p_sys->sofa[i_i_sofa] );
            return VLC_EGENERIC;
        }
        int *p_data_delay_r = p_data_delay + i_m_dim;
        for ( int i = 0 ; i < i_m_dim ; i++ )
        {
            *( p_data_delay + i ) = i_Delay[0];
            *( p_data_delay_r + i ) = i_Delay[1];
        }
    }
    else if ( strncmp ( i_data_delay_dim_name, "M", 1 ) )
    {
        msg_Err ( p_filter, "DataDelay not in the required Dimensions IR or MR.");
        CloseSofa( &p_sys->sofa[i_i_sofa] );
        return VLC_EGENERIC;
    }
    else if ( !strncmp ( i_data_delay_dim_name, "M", 1 ) )
    {
        msg_Dbg( p_filter, "DataDelay in Dimension MR");
        i_status = nc_get_var_int( i_ncid, i_data_delay_id, p_data_delay );
        if (i_status != NC_NOERR)
        {

            CloseSofa( &p_sys->sofa[i_i_sofa] );
            return VLC_EGENERIC;
        }
    }
    p_sys->sofa[i_i_sofa].i_m_dim = i_m_dim;
    p_sys->sofa[i_i_sofa].i_n_samples = i_n_samples;
    p_sys->sofa[i_i_sofa].i_ncid = i_ncid;
    nc_close(i_ncid);
    return VLC_SUCCESS;
}

static int GetSpeakerPos ( filter_t *p_filter, float *p_speaker_pos )
{
    uint16_t i_physical_channels = p_filter->fmt_in.audio.i_physical_channels;
    float *p_pos_temp;
    int i_input_nb = aout_FormatNbChannels( &p_filter->fmt_in.audio );
    if ( i_physical_channels & AOUT_CHAN_LFE )
    {
        i_input_nb--;
    }
    switch ( i_physical_channels )
    {
    case AOUT_CHAN_CENTER:  p_pos_temp = (float[1]){ 0 };
                            break;
    case AOUT_CHANS_STEREO:
    case AOUT_CHANS_2_1:    p_pos_temp = (float[2]){ 30 , 330 };
                            break;
    case AOUT_CHANS_3_0:
    case AOUT_CHANS_3_1:    p_pos_temp = (float[3]){ 30 , 330 , 0 };
                            break;
    case AOUT_CHANS_4_0:
    case AOUT_CHANS_4_1:    p_pos_temp = (float[4]){ 30 , 330 , 120 , 240 };
                            break;
    case AOUT_CHANS_5_0:
    case AOUT_CHANS_5_1:
    case ( AOUT_CHANS_5_0_MIDDLE | AOUT_CHAN_LFE ):
    case AOUT_CHANS_5_0_MIDDLE: p_pos_temp = (float[5]){ 30 , 330 , 120 , 240 , 0 };
                                break;
    case AOUT_CHANS_6_0:    p_pos_temp = (float[6]){ 30 , 330 , 90 , 270 , 150 , 210 };
                            break;
    case AOUT_CHANS_7_0:
    case AOUT_CHANS_7_1:    p_pos_temp = (float[7]){ 30 , 330 , 90 , 270 , 150 , 210 , 0 };
                            break;
    case AOUT_CHANS_8_1:    p_pos_temp = (float[8]){ 30 , 330 , 90 , 270 , 150 , 210 , 180 , 0 };
                            break;
    default: return VLC_EGENERIC;
             break;
    }
    memcpy( p_speaker_pos , p_pos_temp , i_input_nb * sizeof( float ) );
    return VLC_SUCCESS;

}

static int MaxDelay ( struct nc_sofa_t *sofa )
{
    int i_max = 0;
    for ( int  i ; i < ( sofa->i_m_dim * 2 ) ; i++ )
    {
        if ( *( sofa->p_data_delay + i ) > i_max )
            i_max = *( sofa->p_data_delay + i) ;
    }
    return i_max;
}

static int CompensateVolume( filter_t *p_filter)
{
    struct filter_sys_t *p_sys = p_filter->p_sys;
    float f_energy = 0;
    vlc_thread_t thread_find_m;
    struct data_findM_t data_find_m;
    int i_m;
    int i_i_sofa_backup = p_sys->i_i_sofa;
    float *p_ir;
    float f_compensate;
    for ( int i = 0 ; i < N_SOFA ; i++ )
    {
        if( p_sys->sofa[i].i_ncid )
        {
            struct nc_sofa_t *p_sofa = &p_sys->sofa[i];
            p_sys->i_i_sofa = i;
            data_find_m.p_sys = p_sys;
            data_find_m.i_azim = 0;
            data_find_m.i_elev = 0;
            data_find_m.f_radius = 1;
            data_find_m.p_m = &i_m;
            if ( vlc_clone( &thread_find_m, (void *)&sofalizer_FindM, (void *)&data_find_m, VLC_THREAD_PRIORITY_LOW ) ) {} ;
            vlc_join( thread_find_m , NULL );
            p_ir = p_sofa->p_data_ir + 2 * i_m * p_sofa->i_n_samples;
            for ( int j = 0 ; j < p_sofa->i_n_samples ; j ++ )
            {
                f_energy += *( p_ir + j ) * *(p_ir + j );
            }
            f_compensate = 256 / ( p_sofa->i_n_samples * sqrt( f_energy ) );
            msg_Dbg( p_filter, "Compensate-factor: %f", f_compensate );
            p_ir = p_sofa->p_data_ir;
            for ( int j = 0 ; j < ( p_sofa->i_n_samples * p_sofa->i_m_dim * 2 ) ; j++ )
            {
                *( p_ir + j ) *= f_compensate;
            }
        }
    }
    p_sys->i_i_sofa = i_i_sofa_backup;
    return VLC_SUCCESS;
}

static void FreeAllSofa ( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys;
    for ( int i = 0 ; i < N_SOFA ; i++)
    {
        if ( p_sys->sofa[i].i_ncid )
        {
            free ( p_sys->sofa[i].p_sp_a );
            free ( p_sys->sofa[i].p_sp_e );
            free ( p_sys->sofa[i].p_sp_r );
            free ( p_sys->sofa[i].p_data_delay );
            free ( p_sys->sofa[i].p_data_ir );
        }
    }
}

static void FreeFilter( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys;
    free ( p_sys->p_delay_l );
    free ( p_sys->p_delay_r );
    free ( p_sys->p_ir_l );
    free ( p_sys->p_ir_r );
    free ( p_sys->p_ringbuffer_l );
    free ( p_sys->p_ringbuffer_r );
    free ( p_sys->p_speaker_pos );
    free ( p_sys );
}

/*****************************************************************************
* Open:
******************************************************************************/

static int Open( vlc_object_t *p_this )
{
    filter_t *p_filter = (filter_t *)p_this;
    filter_sys_t *p_sys = p_filter->p_sys = malloc( sizeof( *p_sys ) );
    if( unlikely( p_sys == NULL ) )
        return VLC_ENOMEM;

    vlc_object_t *p_out = p_filter->p_parent;
    char *c_filename[N_SOFA];
    const char *psz_var_names_filename[N_SOFA] = { "sofalizer-filename1", "sofalizer-filename2", "sofalizer-filename3" };
    for ( int i = 0 ; i < N_SOFA ; i++ )
    {
        c_filename[i] = var_CreateGetStringCommand( p_filter, psz_var_names_filename[i] );
    }
    p_sys->f_rotation   = abs ( ( - (int) var_CreateGetFloat( p_out, "sofalizer-rotation" ) + 720 ) % 360 );
    p_sys->i_i_sofa     = (int) (var_CreateGetFloat ( p_out, "sofalizer-select" ) ) - 1;
    p_sys->i_switch     = (int) ( var_CreateGetFloat ( p_out, "sofalizer-switch" ) );
    p_sys->f_gain       = var_CreateGetFloat( p_out, "sofalizer-gain" );
    p_sys->f_elevation  = var_CreateGetFloat( p_out, "sofalizer-elevation" );
    p_sys->f_radius     = var_CreateGetFloat( p_out, "sofalizer-radius");


    const char *psz_var_names_azimuth_array[N_POSITIONS] = { "sofalizer-pos1-azi" , "sofalizer-pos2-azi", "sofalizer-pos3-azi", "sofalizer-pos4-azi" };
    for ( int i = 0 ; i < N_POSITIONS ; i++ )
    {
        p_sys->i_azimuth_array[i] = ( var_InheritInteger ( p_out, psz_var_names_azimuth_array[i] ) + 720 ) % 360 ;
    }

    const char *psz_var_names_elevation_array[N_POSITIONS] = { "sofalizer-pos1-ele", "sofalizer-pos2-ele", "sofalizer-pos3-ele", "sofalizer-pos4-ele" };
    for ( int i = 0 ; i < N_POSITIONS ; i++ )
    {
        p_sys->i_elevation_array[i] = var_InheritInteger( p_out, psz_var_names_elevation_array[i] ) ;
    }

    int i_samplingrate = 0;
    int i_samplingrate_old = 0;
    int b_found_valid = false;
    p_sys->b_mute = false ;
    p_sys->i_write = 0;

    /* Load Sofa files, check for Sampling Rate and valid Selection in the Preferences */
    for ( int i = 0 ; i < N_SOFA ; i++ )
    {
        if ( LoadSofa ( p_filter, c_filename[i], i , &i_samplingrate) != VLC_SUCCESS )
        {
            msg_Err(p_filter, "Error while loading SOFA file %d: '%s'", i + 1, c_filename[i] );
        }
        else
        {
            msg_Dbg( p_filter , "File %d: '%s' loaded", i + 1 , c_filename[i] );
            if ( !b_found_valid )
            {
                if ( p_sys->sofa[i].i_ncid )
                {
                    i_samplingrate_old = i_samplingrate;
                    b_found_valid = true;
                }
            }
            if ( p_sys->sofa[i].i_ncid && i_samplingrate != i_samplingrate_old )
            {
                msg_Err ( p_filter, " SOFA file %d '%s' with different Sampling Rate. Discarded.", i + 1, c_filename[i] );
                CloseSofa( &p_sys->sofa[i] );
                p_sys->sofa[i].i_ncid = 0;
            }
        }
    }
    if ( !p_sys->sofa[p_sys->i_i_sofa].i_ncid )
    {
        b_found_valid = false;
        for ( int i = 0 ; i < N_SOFA ; i++)
        {
             if ( !b_found_valid && p_sys->sofa[i].i_ncid )
             {
                p_sys->i_i_sofa = i;
                msg_Err ( p_filter, "Selected File from Settings invalid. Use File %d", i + 1 );
                b_found_valid = true;
             }
        }
        if ( !b_found_valid )
        {
            msg_Err ( p_filter, "No valid file found." );
            FreeAllSofa( p_filter );
            free( p_sys );
            return VLC_EGENERIC;
        }
    }

    /* Set Filter Settings and calculate Speaker Positions*/
    p_filter->fmt_in.audio.i_rate = i_samplingrate_old;
    p_filter->fmt_in.audio.i_format = VLC_CODEC_FL32 ;
    p_filter->fmt_out.audio = p_filter->fmt_in.audio;

    p_filter->fmt_out.audio.i_physical_channels = AOUT_CHANS_STEREO; // required for filter output set to stereo
    p_filter->fmt_out.audio.i_original_channels = AOUT_CHANS_STEREO;

    int i_input_nb = aout_FormatNbChannels( &p_filter->fmt_in.audio );
    if ( p_filter->fmt_in.audio.i_physical_channels & AOUT_CHAN_LFE )
    {
        p_sys->b_lfe = true;
        p_sys->i_n_conv = i_input_nb - 1 ;
    }
    else
    {
        p_sys->b_lfe = false;
        p_sys->i_n_conv = i_input_nb ;
    }

    /* Find the minimum size (length of impulse response plus maximal Delay) of the Ringbuffer as power of 2.  */
    int i_n_max = 0;
    int i_n_current;
    int i_n_max_ir = 0;
    for ( int i = 0 ; i < N_SOFA ; i++ )
    {
        if ( p_sys->sofa[i].i_ncid != 0 )
        {
            i_n_current = p_sys->sofa[i].i_n_samples + MaxDelay ( &p_sys->sofa[i] );
            if ( i_n_current > i_n_max )
            {
                i_n_max = i_n_current;
                i_n_max_ir = p_sys->sofa[i].i_n_samples;
            }
        }
    }
    p_sys->i_buffer_length = pow(2, ceil(log( i_n_max )/ log(2) ) ); /* Buffer length as power of 2 */

    /* Allocate Memory for the impulse responses, delays and the ringbuffers */
    p_sys->p_ir_l = malloc( sizeof(float) * i_n_max_ir * p_sys->i_n_conv  ); /* minus LFE */
    p_sys->p_ir_r = malloc( sizeof(float) * i_n_max_ir * p_sys->i_n_conv );
    p_sys->p_delay_l = malloc ( sizeof( int ) * p_sys->i_n_conv );
    p_sys->p_delay_r = malloc ( sizeof( int ) * p_sys->i_n_conv );
    p_sys->p_ringbuffer_l = calloc( p_sys->i_buffer_length * i_input_nb, sizeof( float ) );
    p_sys->p_ringbuffer_r = calloc( p_sys->i_buffer_length * i_input_nb, sizeof( float ) );
    p_sys->p_speaker_pos = malloc( sizeof( float) * p_sys->i_n_conv );

    if ( !p_sys->p_ir_l || !p_sys->p_ir_r || !p_sys->p_delay_l || !p_sys->p_delay_r || !p_sys->p_ringbuffer_l || !p_sys->p_ringbuffer_r || !p_sys->p_speaker_pos )
    {
        FreeAllSofa( p_filter );
        FreeFilter( p_filter );
        return VLC_ENOMEM;
    }

    CompensateVolume ( p_filter );

    /* Get Speaker positions and load the impulse responses into p_ir_l and p_ir_r for the required directions */
    if ( GetSpeakerPos ( p_filter, p_sys->p_speaker_pos ) != VLC_SUCCESS )
    {
        msg_Err (p_filter, "Couldn't get Speaker Positions. Input Channel Configuration not supported. ");
        FreeAllSofa( p_filter );
        FreeFilter( p_filter );
        return VLC_EGENERIC;
    }
    vlc_mutex_init( &p_sys->lock );
    if ( LoadIR ( p_filter, p_sys->f_rotation, p_sys->f_elevation, p_sys->f_radius ) != VLC_SUCCESS )
    {
        FreeAllSofa( p_filter );
        FreeFilter( p_filter );
        return VLC_ENOMEM;
    }

    msg_Dbg( p_filter, "Samplerate: %d\n Channels to convolute: %d, Ringbufferlength: %d x %d" ,p_filter->fmt_in.audio.i_rate  , p_sys->i_n_conv, i_input_nb, (int )p_sys->i_buffer_length );

    p_filter->pf_audio_filter = DoWork;

    /* Callbacks can call function LoadIR */
    var_AddCallback( p_out, "sofalizer-gain", GainCallback, p_filter );
    var_AddCallback( p_out, "sofalizer-rotation", RotationCallback, p_filter );
    var_AddCallback( p_out, "sofalizer-elevation", ElevationCallback, p_filter );
    var_AddCallback( p_out, "sofalizer-switch", SwitchCallback, p_filter );
    var_AddCallback( p_out, "sofalizer-select", SelectCallback, p_filter );
    var_AddCallback( p_out, "sofalizer-radius", RadiusCallback, p_filter );

    return VLC_SUCCESS;
}

/*****************************************************************************
* DoWork: Prepares the data structures for the threads and starts them
* sofalizer_Convolute: Writes the samples of the input buffer in the ringbuffer
* and convolutes with the impulse response
******************************************************************************/

static block_t *DoWork( filter_t *p_filter, block_t *p_in_buf )
{
    struct filter_sys_t *p_sys = p_filter->p_sys;
    int i_n_clippings_l = 0;
    int i_n_clippings_r = 0;

    int i_input_nb = aout_FormatNbChannels( &p_filter->fmt_in.audio );
    int i_output_nb = aout_FormatNbChannels( &p_filter->fmt_out.audio );

    size_t i_out_size = p_in_buf->i_buffer * i_output_nb / i_input_nb;
    block_t *p_out_buf = block_Alloc( i_out_size );
    if ( unlikely( !p_out_buf ) )
    {
        msg_Warn( p_filter, "can't get output buffer" );
        block_Release( p_in_buf );
        goto out;
    }
    p_out_buf->i_nb_samples = p_in_buf->i_nb_samples;
    p_out_buf->i_dts        = p_in_buf->i_dts;
    p_out_buf->i_pts        = p_in_buf->i_pts;
    p_out_buf->i_length     = p_in_buf->i_length;

    vlc_thread_t left_thread, right_thread;
    struct t_thread_data t_data_l, t_data_r;

    float f_gain_lfe = exp( (p_sys->f_gain - 3 * i_input_nb - 6) / 20 * log(10)); /* -6 dB to get LFE on a similar level */

    t_data_l.p_sys = t_data_r.p_sys = p_sys;
    t_data_l.p_in_buf = t_data_r.p_in_buf = p_in_buf;
    t_data_l.p_input_nb = t_data_r.p_input_nb = &i_input_nb;
    t_data_l.f_gain_lfe = t_data_r.f_gain_lfe = f_gain_lfe;
    t_data_l.p_input_nb = t_data_r.p_input_nb = &i_input_nb;
    t_data_l.f_gain_lfe = t_data_r.f_gain_lfe = f_gain_lfe;
    t_data_l.i_write = t_data_r.i_write = p_sys->i_write;
    t_data_l.p_ringbuffer = p_sys->p_ringbuffer_l;
    t_data_r.p_ringbuffer = p_sys->p_ringbuffer_r;
    t_data_l.p_ir = p_sys->p_ir_l;
    t_data_r.p_ir = p_sys->p_ir_r;
    t_data_l.p_n_clippings = &i_n_clippings_l;
    t_data_r.p_n_clippings = &i_n_clippings_r;
    t_data_l.p_dest = (float *)p_out_buf->p_buffer;
    t_data_r.p_dest = (float *)p_out_buf->p_buffer + 1;
    t_data_l.p_delay = p_sys->p_delay_l;
    t_data_r.p_delay = p_sys->p_delay_r;

    if ( p_sys->b_mute )
    {
        memset( (float *)p_out_buf->p_buffer , 0 , sizeof( float ) * p_in_buf->i_nb_samples * 2 );
    }
    else
    {
        if( vlc_clone( &left_thread, (void *)&sofalizer_Convolute, (void *)&t_data_l, VLC_THREAD_PRIORITY_HIGHEST ) ) goto out;
        if( vlc_clone( &right_thread, (void *)&sofalizer_Convolute, (void *)&t_data_r, VLC_THREAD_PRIORITY_HIGHEST ) ) goto out;
        vlc_join ( left_thread, NULL );
        vlc_join ( right_thread, NULL );
        p_sys->i_write = t_data_l.i_write;
    }

    if ( ( i_n_clippings_l + i_n_clippings_r ) > 0 )
    {
        msg_Err(p_filter, "%d of %d Samples in the Outputbuffer clipped. Please reduce gain.", i_n_clippings_l + i_n_clippings_r, p_out_buf->i_nb_samples * 2 );
    }
out: block_Release( p_in_buf );
    return p_out_buf;
}

void sofalizer_Convolute ( void *p_ptr )
{
    struct t_thread_data *t_data;
    t_data = (struct t_thread_data *)p_ptr;
    struct filter_sys_t *p_sys = t_data->p_sys;
    int i_n_samples = p_sys->sofa[p_sys->i_i_sofa].i_n_samples;
    float *p_src = (float *)t_data->p_in_buf->p_buffer;
    float *p_temp_ir;
    float *p_dest = t_data->p_dest;
    int i_read;
    int *p_delay = t_data->p_delay;
    int i_input_nb = *t_data->p_input_nb;
    int i_buffer_length = p_sys->i_buffer_length;
    uint32_t i_modulo = (uint32_t) i_buffer_length -1 ; /* -1 for AND instead MODULO */
    float *p_ringbuffer[i_input_nb];
    for ( int l = 0 ; l < i_input_nb ; l++ )
    {
        p_ringbuffer[l] = t_data->p_ringbuffer + l * i_buffer_length ;
    }
    int i_write = t_data->i_write;
    float *p_ir = t_data->p_ir;

    for ( int i = t_data->p_in_buf->i_nb_samples ; i-- ; )
    {
        *( p_dest ) = 0;
        for ( int l = 0 ; l < i_input_nb ; l++ )
        {
            *( p_ringbuffer[l] + i_write ) = *( p_src++);
        }
        p_temp_ir = p_ir;
        for ( int l = 0 ; l < p_sys->i_n_conv ; l++ )
        {
            i_read = ( i_write - *( p_delay + l )- (i_n_samples - 1 )  + i_buffer_length ) & i_modulo ;
            for ( int j = i_n_samples ; j-- ; )
            {
                *( p_dest ) += *( p_ringbuffer[l] + ( ( i_read++ ) & i_modulo ) ) * *( p_temp_ir++ );
            }
        }
        if ( p_sys->b_lfe )
        {
            *( p_dest ) += *( p_ringbuffer[p_sys->i_n_conv] + i_write ) * t_data->f_gain_lfe;
        }
        if ( *( p_dest ) >= 1 )
        {
            *t_data->p_n_clippings = *t_data->p_n_clippings + 1;
        }
        p_dest   += 2;
        i_write  = ( i_write + 1 ) & i_modulo ;
    }
    t_data->i_write = i_write;
    return;
}

/*****************************************************************************
* LoadIR: Load the impulse responses (reversed) for directions in p_ir_l and
*     p_ir_r and put the applies the gain to them.
*     Find the correct impulse response with FindM threads.
* FindM:
******************************************************************************/

static int LoadIR ( filter_t *p_filter, int i_azim, int i_elev, float f_radius)
{
    struct filter_sys_t *p_sys = p_filter->p_sys;
    vlc_thread_t thread_find_m[p_sys->i_n_conv];
    struct data_findM_t data_find_m[p_sys->i_n_conv];
    int i_n_samples = p_sys->sofa[p_sys->i_i_sofa].i_n_samples;
    int i_n_conv = p_sys->i_n_conv;
    int i_delay_l[i_n_conv];
    int i_delay_r[i_n_conv];
    int i_input_nb = aout_FormatNbChannels( &p_filter->fmt_in.audio );
    float f_gain_lin = exp( (p_sys->f_gain - 3 * i_input_nb) / 20 * log(10)); /* -3dB per channel */

    float p_ir_l[p_sys->i_n_conv][p_sys->sofa[p_sys->i_i_sofa].i_n_samples];
    float p_ir_r[p_sys->i_n_conv][p_sys->sofa[p_sys->i_i_sofa].i_n_samples];

    int i_m[p_sys->i_n_conv];
    if ( p_sys->i_switch )
    {
        i_azim = p_sys->i_azimuth_array[p_sys->i_switch - 1];
        i_elev = p_sys->i_elevation_array[p_sys->i_switch -1];
    }
    for ( int i = 0 ; i < p_sys->i_n_conv ; i++ )
    {
        data_find_m[i].p_sys = p_sys;
        data_find_m[i].i_azim = (int)(p_sys->p_speaker_pos[i] + i_azim ) % 360;
        data_find_m[i].i_elev = i_elev;
        data_find_m[i].f_radius = f_radius;
        data_find_m[i].p_m = &i_m[i];
        if ( vlc_clone ( &thread_find_m[i] , (void *)&sofalizer_FindM, (void *)&data_find_m[i],  VLC_THREAD_PRIORITY_LOW ) ) {}
    }
    for ( int i = 0 ; i < p_sys->i_n_conv ; i++ )
    {
        vlc_join( thread_find_m[i] , NULL );
        for ( int j = 0 ; j < i_n_samples ; j++ )
        {
            p_ir_l[i][j] = *( p_sys->sofa[p_sys->i_i_sofa].p_data_ir + 2 * i_m[i] * i_n_samples + i_n_samples - 1 - j ) * f_gain_lin;
            p_ir_r[i][j] = *( p_sys->sofa[p_sys->i_i_sofa].p_data_ir + 2 * i_m[i] * i_n_samples + i_n_samples - 1 - j  + i_n_samples ) * f_gain_lin;

        }
        msg_Dbg( p_filter, "Index: %d, Azimuth: %f, Elevation: %f, Radius: %f of SOFA file.", i_m[i], *(p_sys->sofa[p_sys->i_i_sofa].p_sp_a + i_m[i]), *(p_sys->sofa[p_sys->i_i_sofa].p_sp_e + i_m[i]), *(p_sys->sofa[p_sys->i_i_sofa].p_sp_r + i_m[i]) );

        i_delay_l[i] = *( p_sys->sofa[p_sys->i_i_sofa].p_data_delay + 2 * i_m[i] );
        i_delay_r[i] = *( p_sys->sofa[p_sys->i_i_sofa].p_data_delay + 2 * i_m[i] + 1);
    }

    vlc_mutex_lock( &p_sys->lock );
    memcpy ( p_sys->p_ir_l , &p_ir_l[0][0] , sizeof( float ) * p_sys->i_n_conv * p_sys->sofa[p_sys->i_i_sofa].i_n_samples );
    memcpy ( p_sys->p_ir_r , &p_ir_r[0][0] , sizeof( float ) * p_sys->i_n_conv * p_sys->sofa[p_sys->i_i_sofa].i_n_samples );
    memcpy ( p_sys->p_delay_l , &i_delay_l[0] , sizeof( int ) * p_sys->i_n_conv );
    memcpy ( p_sys->p_delay_r , &i_delay_r[0] , sizeof( int ) * p_sys->i_n_conv );
    vlc_mutex_unlock( &p_sys->lock );
    return VLC_SUCCESS;
}

void sofalizer_FindM ( void *p_ptr)
{
    struct data_findM_t *t_data;
    t_data = (struct data_findM_t *)p_ptr;
    struct filter_sys_t *p_sys = t_data->p_sys;
    int i_azim = t_data->i_azim;
    int i_elev = t_data->i_elev;
    float f_radius = t_data->f_radius;
    int *p_m = t_data->p_m;
    float *p_sp_a = p_sys->sofa[p_sys->i_i_sofa].p_sp_a;
    float *p_sp_e = p_sys->sofa[p_sys->i_i_sofa].p_sp_e;
    float *p_sp_r = p_sys->sofa[p_sys->i_i_sofa].p_sp_r;
    int i_m_dim = p_sys->sofa[p_sys->i_i_sofa].i_m_dim;
    int i_i_best = 0;
    float f_delta = 1000;
    float f_current;
    for ( int i = 0; i < i_m_dim ; i++ )
    {
        f_current = fabs ( *(p_sp_a++) - i_azim ) + fabs( *(p_sp_e++) - i_elev ) +  fabs( *(p_sp_r++) - f_radius );
        if ( f_current <= f_delta )
        {
                f_delta = f_current;
                i_i_best = i;
        }
    }
    *p_m = i_i_best;
    return;
}

/*****************************************************************************
* Callbacks
******************************************************************************/

static int GainCallback( vlc_object_t *p_this, char const *psz_var,
                          vlc_value_t oldval, vlc_value_t newval, void *p_data )
{
    VLC_UNUSED(p_this); VLC_UNUSED(psz_var); VLC_UNUSED(oldval);
    filter_t *p_filter = (filter_t *)p_data;
    filter_sys_t *p_sys = p_filter->p_sys;
    vlc_mutex_lock( &p_sys->lock );
    p_sys->f_gain = newval.f_float;
    vlc_mutex_unlock( &p_sys->lock );
    LoadIR( p_filter, p_sys->f_rotation, p_sys->f_elevation, p_sys->f_radius );
    msg_Dbg( p_this , "New Gain-value: %f", newval.f_float );
    return VLC_SUCCESS;
}

static int RotationCallback( vlc_object_t *p_this, char const *psz_var,
                          vlc_value_t oldval, vlc_value_t newval, void *p_data)
{
    VLC_UNUSED(p_this); VLC_UNUSED(psz_var); VLC_UNUSED(oldval);
    filter_t *p_filter = (filter_t *)p_data;
    filter_sys_t *p_sys = p_filter->p_sys;
    float f_temp= (int) (- newval.f_float + 720 ) % 360  ;
    vlc_mutex_lock( &p_sys->lock );
    p_sys->f_rotation = f_temp ;
    vlc_mutex_unlock( &p_sys->lock );
    LoadIR( p_filter, f_temp, p_sys->f_elevation, p_sys->f_radius ) ;
    msg_Dbg( p_filter, "New azimuth-value: %f", f_temp  );
    return VLC_SUCCESS;
}

static int ElevationCallback( vlc_object_t *p_this, char const *psz_var,
                          vlc_value_t oldval, vlc_value_t newval, void *p_data )
{
    VLC_UNUSED(p_this); VLC_UNUSED(psz_var); VLC_UNUSED(oldval);
    filter_t *p_filter = (filter_t *)p_data;
    filter_sys_t *p_sys = p_filter->p_sys;
    vlc_mutex_lock( &p_sys->lock );
    p_sys->f_elevation = newval.f_float ;
    vlc_mutex_unlock( &p_sys->lock ) ;
    LoadIR( p_filter, p_sys->f_rotation, newval.f_float, p_sys->f_radius );
    msg_Dbg( p_filter, "New elevation-value: %f", newval.f_float );
    return VLC_SUCCESS;
}

static int RadiusCallback( vlc_object_t *p_this, char const *psz_var,
                          vlc_value_t oldval, vlc_value_t newval, void *p_data )
{
    VLC_UNUSED(p_this); VLC_UNUSED(psz_var); VLC_UNUSED(oldval);
    filter_t *p_filter = (filter_t *)p_data;
    filter_sys_t *p_sys = p_filter->p_sys;
    vlc_mutex_lock( &p_sys->lock );
    p_sys->f_radius = newval.f_float ;
    vlc_mutex_unlock( &p_sys->lock ) ;
    LoadIR( p_filter, p_sys->f_rotation, p_sys->f_elevation,  newval.f_float );
    msg_Dbg( p_filter, "New radius-value: %f", newval.f_float );
    return VLC_SUCCESS;
}


static int SwitchCallback( vlc_object_t *p_this, char const *psz_var,
                          vlc_value_t oldval, vlc_value_t newval, void *p_data )
{
    VLC_UNUSED(p_this); VLC_UNUSED(psz_var); VLC_UNUSED(oldval);
    filter_t *p_filter = (filter_t *)p_data;
    filter_sys_t *p_sys = p_filter->p_sys;
    vlc_mutex_lock( &p_sys->lock );
    p_sys->i_switch = (int) newval.f_float ;
    if ( p_sys->i_switch )
    {
        for ( int i = 0 ; i < p_sys->i_n_conv ; i++ ) *(p_sys->p_speaker_pos + i ) = 0;
    }
    else
    {
        GetSpeakerPos ( p_filter, p_sys->p_speaker_pos );
    }
    vlc_mutex_unlock( &p_sys->lock ) ;
    LoadIR ( p_filter, p_sys->f_rotation, p_sys->f_elevation, p_sys->f_radius );
    msg_Dbg( p_filter, "New Switch-Position: %d", (int) newval.f_float );
    return VLC_SUCCESS;
}

static int SelectCallback( vlc_object_t *p_this, char const *psz_var,
                          vlc_value_t oldval, vlc_value_t newval, void *p_data )
{
    VLC_UNUSED(p_this); VLC_UNUSED(psz_var); VLC_UNUSED(oldval);
    filter_t *p_filter = (filter_t *)p_data;
    filter_sys_t *p_sys = p_filter->p_sys;
    vlc_mutex_lock( &p_sys->lock );
    if ( p_sys->sofa[((int)newval.f_float + 5 - 1 ) % 5].i_ncid )
    {
        p_sys->i_i_sofa = ( (int) newval.f_float + 5 - 1) % 5 ;
        p_sys->b_mute = false;
        vlc_mutex_unlock( &p_sys->lock ) ;
        LoadIR ( p_filter, p_sys->f_rotation, p_sys->f_elevation , p_sys->f_radius );
        msg_Dbg( p_filter, "New Sofa-Select: %f", newval.f_float );
    }
    else
    {
        msg_Dbg( p_filter, "Invalid File selected!" );
        p_sys->b_mute = true;
        vlc_mutex_unlock( &p_sys->lock ) ;
    }
    return VLC_SUCCESS;
}

/*****************************************************************************
* Close:
******************************************************************************/

static void Close( vlc_object_t *p_this )
{
    filter_t *p_filter = ( filter_t* )p_this;
    filter_sys_t *p_sys = p_filter->p_sys;
    vlc_object_t *p_out = p_filter->p_parent;

    var_DelCallback( p_out, "sofalizer-gain", GainCallback, p_filter );
    var_DelCallback( p_out, "sofalizer-rotation", RotationCallback, p_filter );
    var_DelCallback( p_out, "sofalizer-elevation", ElevationCallback, p_filter );
    var_DelCallback( p_out, "sofalizer-switch", SwitchCallback, p_filter );
    var_DelCallback( p_out, "sofalizer-select", SelectCallback, p_filter );
    var_DelCallback( p_out, "sofalizer-radius", RadiusCallback, p_filter );

    vlc_mutex_destroy( &p_sys->lock );

    FreeAllSofa( p_filter );
    FreeFilter( p_filter );
}
